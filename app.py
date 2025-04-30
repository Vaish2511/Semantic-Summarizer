import os
import re
import streamlit as st
from docx import Document
import PyPDF2
import torch
from transformers import (
    pipeline,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging

# ─── LOGGING SETUP ───────────────────────────────────────────
# Configuring logging to capture events, warnings, and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── UTILITY CLASSES ─────────────────────────────────────────

class FileReader:
    """
    This class is responsible for reading different file types: PDF, DOCX, and TXT.
    Each method returns the cleaned text from the respective file format.
    """
    @staticmethod
    def read_docx(f):
        """Reads DOCX file and returns the text."""
        doc = Document(f)
        return "\n".join(p.text for p in doc.paragraphs)

    @staticmethod
    def read_pdf(f):
        """Reads PDF file and returns the text."""
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def read_txt(f):
        """Reads TXT file and returns the text."""
        return f.read().decode("utf-8")


class TextProcessor:
    """
    This class is responsible for text preprocessing:
    - Cleaning unwanted sections (e.g., references, tables)
    - Tokenizing the text into sentences and words
    """
    @staticmethod
    def clean_text(text):
        """
        Cleans text by removing unwanted sections like references and tables, and
        handling broken hyphens.
        """
        txt = text.replace("\n", " ")
        # Glue broken hyphens (e.g., "long-" and "text" becomes "longtext")
        txt = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", txt)
        # Drop sections like 'Acknowledgment' or 'References'
        lower = txt.lower()
        for marker in ("acknowledgment", "acknowledgements", "references"):
            if marker in lower:
                txt = txt[: lower.index(marker)]
        return txt.strip()

    @staticmethod
    def remove_redundant_sentences(text: str):
        """
        Removes duplicate sentences from the text using regex to tokenize sentences.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        seen = set()
        output = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                output.append(s)
        return " ".join(output)

    @staticmethod
    def smart_split_sentences(text, max_sentences=5):
        """
        Splits text into chunks, where each chunk contains a maximum number of sentences.
        This ensures that the text remains within model input limits.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def word_tokenize(text):
        """
        Tokenizes text into words using regex, similar to nltk.word_tokenize.
        """
        return re.findall(r'\b\w+\b', text.lower())  # Extract words by matching word boundaries


class Summarizer:
    """
    Singleton class to load and cache the summarization models, ensuring models are only loaded once.
    Handles chunked summarization of the documents.
    """
    _instance = None

    def __new__(cls, model_choice):
        """Ensures that only one instance of the Summarizer class exists."""
        if not cls._instance:
            cls._instance = super(Summarizer, cls).__new__(cls)
            cls._instance.model_choice = model_choice
            cls._instance.summarizer_pipeline, cls._instance.tokenizer = cls.load_model_pipeline(model_choice)
        return cls._instance

    @staticmethod
    def load_model_pipeline(choice):
        """
        Loads the pre-trained model based on the user's choice.
        Models are loaded once and cached for subsequent requests.
        """
        logger.info(f"Loading model: {choice}")

        # Load the tokenizer and model based on the selected option
        if choice == "DistilBART":
            tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6", use_fast=True)
            model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
        elif choice == "T5-Small":
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
        elif choice == "T5-Base":
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            model = T5ForConditionalGeneration.from_pretrained("t5-base")
        else:
            raise ValueError(f"Model choice {choice} is not recognized.")

        # Initialize the summarization pipeline
        summarizer_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logger.info(f"Model {choice} loaded successfully.")
        return summarizer_pipeline, tokenizer

    def build_few_shot_prompt(self, new_input):
        """
        Builds a few-shot prompt for the model, guiding it on how to summarize the given text.
        """
        prompt = f"Summarize this technical paper:\n{new_input}\nSummary:"
        return prompt

    def summarize_text_few_shot(self, text):
        """
        Summarizes the provided text by splitting it into chunks and processing them in parallel.
        """
        chunks = TextProcessor.smart_split_sentences(text, max_sentences=5)
        summaries = []

        # Use ThreadPoolExecutor for parallel processing of chunks
        with ThreadPoolExecutor() as executor:
            future_to_chunk = {executor.submit(self.summarize_chunk, chunk): chunk for chunk in chunks}
            for idx, future in enumerate(future_to_chunk):
                try:
                    result = future.result()
                    summaries.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    st.warning(f"Chunk failed: {e}")

        # Combine all chunk summaries into one final summary
        final_summary = " ".join(summaries)
        return final_summary

    def summarize_chunk(self, chunk):
        """Summarizes a single chunk of text."""
        prompt = self.build_few_shot_prompt(chunk)
        try:
            result = self.summarizer_pipeline(prompt, max_length=150, do_sample=True)[0]
            return result['summary_text'].strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return f"Error summarizing chunk: {e}"

    def extractive_top_k(self, text: str, k: int=3):
        """
        Extracts the top K most relevant sentences based on semantic similarity.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)  # Tokenizing sentences with regex
        if len(sentences) <= k:
            return sentences
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sent_embs = model.encode(sentences)
        doc_emb = model.encode([text])[0]
        # Calculate cosine similarity
        sims = (sent_embs @ doc_emb) / (
            (sent_embs**2).sum(axis=1)**0.5 * (doc_emb**2).sum()**0.5
        )
        # Sort sentences by similarity and select the top k
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        # Sort the selected top sentences by their original order
        top_idx.sort()
        return [sentences[i] for i in top_idx]

class Visualization:
    """
    Singleton class to handle various visualizations for the document comparison:
    - Word frequency heatmap
    - Sentence embedding scatter plot
    """
    _instance = None

    def __new__(cls):
        """Ensures that only one instance of the Visualization class exists."""
        if not cls._instance:
            cls._instance = super(Visualization, cls).__new__(cls)
        return cls._instance

    def plot_frequency_heatmap(self, sum1:str, sum2:str):
        """Generates a heatmap for the word frequency comparison between two summaries."""
        cnt1 = Counter(TextProcessor.word_tokenize(sum1))
        cnt2 = Counter(TextProcessor.word_tokenize(sum2))
        
        # Combine top 20 words from both summaries
        topw = set([w for w, _ in cnt1.most_common(20)] + [w for w, _ in cnt2.most_common(20)])
        
        df = pd.DataFrame({
            "Summary": ["Full"] * len(topw) + ["Extractive"] * len(topw),
            "Word": list(topw) * 2,
            "Freq": [cnt1[w] for w in topw] + [cnt2[w] for w in topw],
        })
        
        # Use pivot_table for proper handling of data
        pivot = df.pivot_table(index="Word", columns="Summary", values="Freq", aggfunc="sum").fillna(0)
        
        # Plot using Plotly
        fig = px.imshow(
            pivot,
            labels=dict(x="Summary", y="Word", color="Count"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_embedding_scatter(self, sum1:str, sum2:str):
        """Generates a PCA scatter plot of the sentence embeddings of the two summaries."""
        sents1 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', sum1)
        sents2 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', sum2)
        labels = ["Full"] * len(sents1) + ["Extractive"] * len(sents2)
        sentences = sents1 + sents2
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(sentences)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        df = pd.DataFrame(coords, columns=["x", "y"])
        df["Summary"] = labels
        df["Sentence"] = sentences
        fig = px.scatter(
            df, x="x", y="y", color="Summary",
            hover_data=["Sentence"],
            title="Sentence Embedding PCA"
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── STREAMLIT UI ───────────────────────────────────────────
st.set_page_config(page_title="Semantic Summarizer", layout="wide")
st.title("📄 Semantic Summarizer")

# Sidebar
st.sidebar.header("Upload Documents")
f1 = st.sidebar.file_uploader("First doc", type=["pdf","docx","txt"])
f2 = st.sidebar.file_uploader("Second doc", type=["pdf","docx","txt"])

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Summarization Model", ("DistilBART", "T5-Small", "T5-Base"))
start = st.sidebar.button("Summarize")

if start:
    if not (f1 and f2):
        st.sidebar.error("Please upload both files.")
        st.stop()

    ext1 = f1.name.rsplit(".",1)[-1].lower()
    ext2 = f2.name.rsplit(".",1)[-1].lower()
    raw1 = FileReader.read_pdf(f1) if ext1 == "pdf" else FileReader.read_docx(f1) if ext1 == "docx" else FileReader.read_txt(f1)
    raw2 = FileReader.read_pdf(f2) if ext2 == "pdf" else FileReader.read_docx(f2) if ext2 == "docx" else FileReader.read_txt(f2)
    text1, text2 = TextProcessor.clean_text(raw1), TextProcessor.clean_text(raw2)

    # Generative Summary
    summarizer = Summarizer(model_choice)
    with st.spinner("Generating summaries…"):
        gen1 = summarizer.summarize_text_few_shot(text1)
        gen2 = summarizer.summarize_text_few_shot(text2)

    # Clean up Repetition
    full1 = TextProcessor.remove_redundant_sentences(gen1)
    full2 = TextProcessor.remove_redundant_sentences(gen2)

    # Extract top 3 relevant sentences
    top3_1 = summarizer.extractive_top_k(text1, k=3)
    top3_2 = summarizer.extractive_top_k(text2, k=3)

    # Calculate similarity and dissimilarity
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode([full1, full2])
    similarity_score = (embeddings[0] @ embeddings[1]) / (
        (embeddings[0]**2).sum()**0.5 * (embeddings[1]**2).sum()**0.5
    )
    overlap_score = len(set(full1.lower().split()).intersection(set(full2.lower().split()))) / len(set(full1.lower().split()).union(set(full2.lower().split())))
    dissimilarity_score = 1 - similarity_score

    st.success("✅ Done!")

    # Metrics
    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Similarity Score", f"{similarity_score:.2f}")
    
    with col2:
        st.metric("Overlap %", f"{overlap_score * 100:.2f}%")
    
    with col3:
        st.metric("Dissimilarity Score", f"{dissimilarity_score:.2f}")

    # Comparison
    st.subheader("Summary Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.text_area("First Summary", full1, height=200)
        st.text_area("First Top 3 Extractive", "\n".join(top3_1), height=150)

    with col2:
        st.text_area("Second Summary", full2, height=200)
        st.text_area("Second Top 3 Extractive", "\n".join(top3_2), height=150)

    # Interactive Visuals
    st.subheader("Interactive Visualizations")
    st.markdown("**1. Word Frequency Heatmap**")
    Visualization().plot_frequency_heatmap(full1, full2)

    st.markdown("**2. Sentence Embedding Scatter**")
    Visualization().plot_embedding_scatter(full1, full2)