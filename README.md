## Semantic Summarizer: Document Summarization and Similarity Comparison

## Overview:
This project implements a semantic summarization system that can process documents (PDF, DOCX, TXT) and generate concise summaries while respecting the specified summary length. Additionally, it computes similarity metrics between two documents and presents insightful visualizations, all through an intuitive Streamlit interface.

## Key Features:
1. Summarization of technical documents with user-defined summary length.
2. Similarity comparison between two documents using cosine similarity and overlap metrics.
3. Interactive visualizations for similarity, dissimilarity, and overlap metrics.
4. Handles references, tables, and figures to focus on the main content of the document.
5. Support for three models: DistilBART, DistilGPT2, and T5-Small.
6. Page-wise summarization for large documents to generate concise and relevant summaries.
7. Logging to track the flow of execution and capture errors.

## Table of Contents
1. [Project Setup](#1-project-setup)
2. [Dependencies](#2-dependencies)
3. [Usage Instructions](#3-usage-instructions)
4. [Application Details](#4-application-details)
    - i. [Text Preprocessing](#i-text-preprocessing)
    - ii. [Model Selection](#ii-model-selection)
    - iii. [User Interface](#iii-user-interface)
    - iv. [Visualizations](#iv-visualizations)
5. [Logging](#5-logging)
6. [Development Notes](#6-development-notes)
7. [Future Enhancements](#7-future-enhancements)

## **1. Project Setup**
To get the application up and running on your local machine, follow the steps below.
#### Step 1: Clone the Repository
First, clone the repository to your local machine:
> git clone https://github.com/your-repository/semantic-summarizer.git
>> cd semantic-summarizer
#### Step 2: Set Up the Virtual Environment
Create and activate a virtual environment:
> python -m venv venv
>> source venv/bin/activate
#### Step 3: Install Dependencies
Install the required dependencies using the requirements.txt file:
> pip install -r requirements.txt
#### Step 4: Run the Application
After installing the dependencies, you can run the Streamlit app:
> streamlit run app.py

The application will open in your default web browser, and you can interact with it via the Streamlit UI.

## **2. Dependencies**
The application relies on the following libraries:
1. Streamlit: For building the interactive web interface.
2. Transformers: For loading the pre-trained models for summarization.
3. SentenceTransformers: For calculating cosine similarity and overlap between summaries.
4. PyPDF2, python-docx: For reading PDF and DOCX files.
5. matplotlib, seaborn: For generating visualizations of the metrics.
6. Logging: For tracking the flow and errors of the application.

Here’s the list of dependencies in requirements.txt:
1. streamlit==1.14.0
2. docx==0.2.5
3. PyPDF2==1.26.0
4. matplotlib==3.5.1
5. seaborn==0.11.2
6. scikit-learn==1.0.2
7. sentence-transformers==2.2.0
8. torch==1.10.0
9. transformers==4.11.3

## **3. Usage Instructions**
1. Upload Documents: Use the Sidebar to upload two documents (PDF, DOCX, or TXT) for comparison.
2. Select the Model: Choose between the three models: DistilBART, T5 Base and T5 Small.
3. Start Summarization: Click on the Summarize button to process the documents.
4. View Summary and Metrics: View the summaries for both documents. Compare the similarity, overlap, and dissimilarity metrics. Visualize other metrics with the heatmap charts.

## **4. Application Details**
### **i. Text Preprocessing**
The input text is cleaned to remove unnecessary sections, including:
1. References (e.g., "[1], [Table 1]")
2. Figures and Tables (e.g., "Figure 1", "Table 2")
3. Headers and footers that may appear in scanned documents or unstructured text.
The system splits the text into chunks (sentences or paragraphs), and each chunk is processed for summarization.
    
### **ii. Model Selection**
The application currently supports the following models:
1. DistilBART: A smaller, distilled version of BART designed for summarization.
2. T5-Base: A transformer-based model designed for text generation and summarization tasks, which works well for general text summarization.
3. T5-Small: A smaller variant of T5 designed for efficient summarization.
The user can select one of these models in the sidebar. The models will be loaded once, and the summarization will proceed chunk by chunk.

### **iii. User Interface***
The Streamlit interface is designed to be intuitive:
1. File Upload: Allows users to upload two documents.
2. Model Choice: Users can choose the exact model they want to use for the summarization task.
3. Summarize Button: Triggers the summarization process for both documents.
4. Side-by-Side Comparison: Displays the summaries of both documents with visualizations and metrics.

### **iv. Visualizations**
The application provides the following visualizations for a detailed comparison:
1. Word Frequency Heatmap: A color-coded heatmap that displays the frequency of common words between the two summaries, helping to highlight key terms in both documents.
2. Sentence Embedding Scatter Plot: A scatter plot that visualizes the sentence embeddings of both summaries, providing insight into their semantic similarity.
These visualizations help in understanding the relationship between the two documents and their summarized versions.

Example Output:
    Metrics: Similarity Score: 0.85; Overlap: 75%; Dissimilarity: 0.15

## **5. Logging**
The application uses Python's built-in logging module to track key events:
1. Model Loading: Logs when models are loaded and which model is selected.
2. Summarization Progress: Logs the progress of each chunk being summarized.
3. Errors and Warnings: Captures and logs any errors or warnings that occur during the process.
This ensures that if something goes wrong, you can trace it back through the logs for debugging purposes.

## **6. Development Notes**
1. Multi-threading: The summarization process uses ThreadPoolExecutor to handle the summarization of multiple chunks in parallel, improving performance on larger documents.
2. Model Optimization: Currently, models are loaded on CPU due to the absence of GPUs, which may cause slower inference times. For faster inference, consider switching to GPU if available.
3. Token Length Restrictions: Some models have token length limits (e.g., T5, GPT2). The text is chunked and preprocessed to ensure it stays within the model's token limit.

## **7. Future Enhancements**
1. GPU Support: If a GPU is available, the application could utilize it for faster inference by adjusting the device argument in the pipeline.
2. Summarization Quality: Fine-tuning the selected models on specific technical papers could improve the quality of summaries for specialized documents.
3. Better Summary Merging: After summarizing each page, merging the page-wise summaries intelligently to meet the target length could be implemented.
4. User Authentication: Adding authentication (via OAuth or another service) could help in handling large-scale deployments and personalized summaries.

### 8. Conclusion
This Semantic Summarizer provides an intuitive interface for generating summaries of technical documents with customizable lengths. It also supports similarity comparison between two documents and visualizes the results for better insight. The system is designed to be efficient and scalable, with the ability to handle large documents and ensure production-quality output through logging and error tracking.
