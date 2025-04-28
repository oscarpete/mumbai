# Pydantic AI Expert System for Indian Building Code

This project implements an AI-powered expert system focused on the National Building Code of India 2016. It ingests the code from PDF documents, processes them, and provides an interface (likely via Streamlit) to query or interact with the information.

## Project Structure

```
.
├── .env                  # Environment variables (API keys, configurations)
├── convert_pdf_to_md.py  # Script to convert PDF documents to Markdown
├── ingest_markdown_docs.py # Script to ingest Markdown files into a data store
├── ingest_web_docs.py    # Script to ingest documents from web URLs
├── ingestion_utils.py    # Utility functions for data ingestion processes
├── pydantic_ai_expert.py # Core logic for the AI expert system, potentially using Pydantic
├── requirements.txt      # Python dependencies for the project
├── site_pages.sql        # SQL file, possibly for database schema or web page data
├── streamlit_ui.py       # Streamlit application for the user interface
├── data/                 # Directory containing processed data (Markdown files)
│   ├── National Building Code of India 2016 Volume 1.pages_....md
│   └── National Building Code of India 2016 Volume 2.pages_....md
└── README.md             # This file
```

## Components

1.  **Data Ingestion**:
    *   `convert_pdf_to_md.py`: Converts the source PDF files (National Building Code of India) into Markdown format, stored in the `data/` directory.
    *   `ingest_markdown_docs.py`: Takes the generated Markdown files and likely loads them into a vector database or other storage suitable for retrieval.
    *   `ingest_web_docs.py`: Potentially ingests related information from web sources.
    *   `ingestion_utils.py`: Contains helper functions used by the ingestion scripts.
2.  **AI Expert Core**:
    *   `pydantic_ai_expert.py`: Contains the main logic for processing queries and retrieving relevant information from the ingested data, possibly using language models and Pydantic for structuring input/output.
3.  **User Interface**:
    *   `streamlit_ui.py`: Provides a web-based user interface using Streamlit, allowing users to interact with the expert system.

## Setup and Usage

1.  **Environment Variables**: Create a `.env` file based on a potential `.env.example` (if provided) or configure necessary variables like API keys.
2.  **Dependencies**: Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Preparation**:
    *   Run `convert_pdf_to_md.py` if the source PDFs are available and need conversion.
    *   Run `ingest_markdown_docs.py` to load the Markdown data into the system's knowledge base.
    *   (Optional) Run `ingest_web_docs.py` if web data ingestion is needed.
4.  **Run the Application**:
    ```bash
    streamlit run streamlit_ui.py
    ```

This will start the Streamlit web server, and you can access the application through your browser.
