# Document Assistant Using RAG

A Retrieval-Augmented Generation (RAG) system for intelligent document querying and answering, built with Python and modern NLP technologies.

## Overview

This project implements a RAG system that allows users to upload documents and ask questions about their content. The system processes documents, indexes their content, and generates accurate answers based on the relevant context.

### Key Features

- Document processing and intelligent chunking
- Vector-based semantic search for accurate information retrieval
- Advanced response generation using Mistral-7B
- Response validation system
- Asynchronous processing for better performance
- REST API interface
- Docker containerization

## Architecture

The system is built on a modular architecture with several key components:

- **AgentManager**: Orchestrates the entire RAG pipeline
- **IndexingAgent**: Handles document processing and vectorization
- **ResearchAgent**: Manages context retrieval
- **QAAgent**: Generates responses using LLM
- **ValidationAgent**: Ensures response quality
- **Vector Store**: SQLite-based vector storage with connection pooling

## Requirements

- Python 3.9+
- Docker and Docker Compose
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional but recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-document-assistant.git
cd rag-document-assistant
```

2. Build the Docker image:
```bash
docker-compose build
```

3. Start the service:
```bash
docker-compose up -d
```

## Configuration

The system can be configured through `config/config.yaml`. Main configuration options:

```yaml
model_config:
  model_name: "mistralai/Mistral-7B-v0.1"
  device: "cuda"  # or "cpu"
  max_length: 1024
  temperature: 0.7
  
vector_store:
  type: "sqlite"
  settings:
    chunk_size: 512
    similarity_threshold: 0.7
    max_results: 5
```

## Usage

### REST API

The system exposes two main endpoints:

1. Upload a document:
```bash
curl -X POST -F "file=@your_document.txt" http://localhost:8000/upload
```

2. Ask a question:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "What is the main topic of the document?"}' \
     http://localhost:8000/ask
```

### Example

Let's say we have a technical document about Python programming:

1. Upload the document:
```bash
curl -X POST -F "file=@python_guide.txt" http://localhost:8000/upload
```

2. Ask a question:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "What are the key features of Python?"}' \
     http://localhost:8000/ask
```

Response:
```json
{
  "answer": "Python's key features include its simplicity, readability, and extensive library support...",
  "confidence": 0.85,
  "sources": ["Chapter 1: Introduction to Python", "Section 2.1: Python Features"]
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Project Structure
```
rag-document-assistant/
├── src/
│   ├── agents/
│   │   ├── agent_manager.py
│   │   ├── indexing_agent.py
│   │   ├── research_agent.py
│   │   ├── qa_agent.py
│   │   └── validation_agent.py
│   ├── generator/
│   │   └── llm_interface.py
│   └── retriever/
│       └── retrieval.py
├── tests/
├── config/
├── docker/
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.