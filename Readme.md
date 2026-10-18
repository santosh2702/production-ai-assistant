# Production AI Assistant 🤖

A document-based AI assistant built with FastAPI, FAISS, Sentence Transformers, and Groq. Upload a PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).

## Features
- 📄 PDF upload and text extraction
- 🔍 Semantic search using FAISS
- 🧠 Context-aware answers using Groq LLMs
- ⚡ FastAPI REST APIs
- 🛡️ Input validation and rate limiting
- 🐳 Dockerized for deployment

## Tech Stack
- Python, FastAPI
- FAISS, Sentence Transformers
- Groq API
- Docker

## Run Locally

```bash
git clone <your-repository-url>
cd production-ai-assistant

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt