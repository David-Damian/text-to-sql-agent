# 🤖 AI-Powered Text-to-SQL Agent

> **A friendly, high-performance conversational agent** that translates natural language questions into executable SQL queries using advanced **RAG (Retrieval-Augmented Generation)** techniques over a mock E-Commerce DuckDB database.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green?logo=langchain)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT_&_Embeddings-412991?logo=openai)
![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory_DB-F0B800)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-blue)

---

## 📋 Table of Contents

- [Business Context](#-business-context)
- [Objective](#-objective)
- [Architecture & Methodology](#-architecture--methodology)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [How to Run](#-how-to-run)
- [Data Privacy & Security](#-data-privacy--security)

---

## 💼 Business Context

Data democratization is crucial for modern businesses. However, business stakeholders often rely on data engineering or BI teams to write complex SQL queries to extract insights. This creates a technical bottleneck and slows down decision-making. 

This project solves that by providing an interface where users can ask questions in **plain English** (e.g., *"What is the total revenue for April?"*), and the AI agent will intelligently formulate and execute the correct SQL query against the database, returning the data instantly.

---

## 🎯 Objective

1. **Schema Retrieval:** Use RAG (Retrieval-Augmented Generation) to ingest and index database schemas so the LLM perfectly understands the available tables and columns.
2. **Text-to-SQL Translation:** Dynamically generate accurate and optimized SQL syntax using OpenAI's models (`ChatOpenAI`).
3. **Execution & Answer:** Execute the generated SQL securely using an in-memory database (`DuckDB`) and return the parsed results directly to the user.

---

## 🔬 Architecture & Methodology

The pipeline follows a state-of-the-art **RAG Architecture** heavily optimized for interacting with databases:

### 1. Database Setup (DuckDB)
- We initialize an in-memory, zero-dependency analytical database modeling an **E-Commerce environment** (Tables include `users`, `orders`, `products`, `reviews`, `promotions`, etc.).

### 2. Knowledge Ingestion & Vectorization
- Database schemas, table relationships, and metadata are converted into searchable `Langchain Documents`.
- We use **OpenAI Embeddings** to vectorize the schema semantics.
- **FAISS (Facebook AI Similarity Search)** is used as the local, high-speed vector store to host these embeddings for blazing-fast retrieval.

### 3. Agentic Workflow
- **Question:** The user asks a natural language question.
- **Retrieval:** The agent sweeps the FAISS index to find the most relevant tables and columns for the specific question provided.
- **Prompting:** The retrieved context is safely injected into a strict `ChatPromptTemplate` advising the LLM on DuckDB's SQL dialect and the available schema.
- **Execution:** The SQL query is generated, executed against the DuckDB instance, and the raw analytical answer is formatted nicely.

---

## 📁 Project Structure

```text
.
├── README.md                          # This file
├── .gitignore                         # Excludes sensitive data (PDFs, JSONs, API keys)
├── notebooks/
│   └── text_to_sql.ipynb              # Main pipeline for RAG and Text-to-SQL generation
└── src/
    └── duckdb.py                      # Initializes the in-memory E-Commerce DB schema
```

> **Note:** Raw datasets (like `.json` databases) and challenge instructions (`.pdf`) are explicitly ignored via `.gitignore` to maintain a clean, lightweight, and secure repository.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM & Agent Logic** | LangChain / OpenAI | Reasoning, prompt routing, and SQL generation |
| **Embeddings** | OpenAI Embeddings | Vectorizing DB schema & metadata |
| **Vector Store** | FAISS | Fast similarity search for schema retrieval |
| **Database Engine** | DuckDB | Blazing fast in-memory analytical SQL database |
| **Environment** | Jupyter Notebooks | Interactive development, steps, and visualization |

---

## 🚀 How to Run

### Prerequisites

Create a virtual environment and install the required dependencies:

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core packages
pip install langchain langchain-openai langchain-community faiss-cpu duckdb pandas jupyter
```

### Environment Variables

You must provide a valid OpenAI API key. Create a `.env` file in the root directory or export it directly:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

*Note: The `.env` file is strictly ignored by Git to prevent accidental exposure of your secrets.*

### Execution

1. **Verify Database:** Run the `duckdb.py` script to explore the mock E-Commerce schema.
   ```bash
   python src/duckdb.py
   ```
2. **Run Pipeline:** Launch Jupyter and open the main notebook to interact with the Text-to-SQL agent.
   ```bash
   jupyter notebook notebooks/text_to_sql.ipynb
   ```

---

## 🔐 Data Privacy & Security

To uphold technical rigor and repository best-practices:
- **No Data Leakage:** Raw backend data like `sql_dataset_bourbaki.json` and challenge instructions like the `NLP Avanzado_ Reto V (1).pdf` are kept fully local and are excluded from version control.
- **No Secrets Leaked:** Environment configurations and API Keys are never pushed to the remote repository.
