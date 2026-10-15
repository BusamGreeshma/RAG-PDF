# RAG-PDF: Chat with Your PDF Using AI

RAG-PDF is a Python-based PDF question-answering application that allows users to upload a PDF document and ask questions about its content through a conversational interface.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant document passages and generate answers based on the available context. It combines document processing, semantic embeddings, vector search, and a locally loaded language model.

## Features

- **PDF Upload:** Upload PDF documents directly through the web interface.
- **Text Extraction:** Extract text from PDF pages using PyPDFLoader.
- **Text Chunking:** Divide extracted content into smaller, overlapping chunks for retrieval.
- **Semantic Embeddings:** Convert text chunks into vector representations using a Hugging Face embedding model.
- **Vector Search:** Use Qdrant for similarity-based retrieval of relevant document passages.
- **AI-Powered Question Answering:** Generate responses using a locally loaded FLAN-T5 model.
- **Conversational Interface:** Ask questions and view responses in a Streamlit chat interface.
- **Local Model Execution:** Designed to run the language model locally rather than relying on a hosted LLM API.

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Document Processing:** LangChain Community, PyPDFLoader
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Database:** Qdrant
- **Language Model:** Google FLAN-T5 Base
- **RAG Pipeline:** LangChain

## How It Works

1. The user uploads a PDF document.
2. PyPDFLoader extracts the document text.
3. RecursiveCharacterTextSplitter divides the text into chunks of 1,000 characters with an overlap of 100 characters.
4. Hugging Face embeddings convert the chunks into numerical vectors.
5. Qdrant stores the vectors in an in-memory vector store.
6. When the user asks a question, the retriever searches for relevant document passages.
7. The retrieved context is passed to the language model.
8. The application displays the generated answer in the chat interface.

## Project Structure

```text
RAG-PDF/
├── PDFproject.py
└── README.md
```

## Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies
- pip
- Internet access for the initial download of required Python packages and pretrained models
- Sufficient RAM and storage for the embedding model and language model

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/BusamGreeshma/RAG-PDF.git
cd RAG-PDF
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install the dependencies

Install the packages used by the application:

```bash
python -m pip install streamlit langchain-community langchain-text-splitters langchain-qdrant langchain-classic sentence-transformers transformers torch pypdf
```

If dependency compatibility issues occur, use a tested requirements file with compatible package versions.

### 4. Run the application

```bash
streamlit run PDFproject.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Usage

1. Launch the application.
2. Upload a PDF file.
3. Wait for the document to be processed.
4. Enter a question related to the uploaded document.
5. Read the answer generated using retrieved document context.
6. Ask additional questions as needed.

## Current Implementation Notes

- The vector store uses in-memory storage and is not persistent between application reruns.
- The document is processed through the application's PDF-upload workflow.
- The quality of responses depends on document text extraction, retrieval relevance, model compatibility, and the available context.
- The application should be tested with different PDF layouts and document sizes.
- The current implementation focuses on text-based PDF question answering; image understanding is not implemented in the reviewed source code.

## Future Enhancements

- Add persistent vector storage.
- Display source passages and page references alongside answers.
- Improve document-processing efficiency by avoiding unnecessary reprocessing.
- Add support for multiple PDFs and document collections.
- Improve error handling and loading indicators.
- Evaluate retrieval quality and answer accuracy on a test set.
- Add support for tables and visual content in PDFs.

## Author

**Greeshma Busam**

GitHub: https://github.com/BusamGreeshma


