# Quick Start Guide

Minimal commands to get started. For detailed documentation, see [README_RAG.md](README_RAG.md).

## Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not done)
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Chunking
python chunking.py --input fr_docs --output chunks.json

# 2. Create vector DB
python vectorize.py --chunks chunks.json --output vector_db

# 3. Query (simple search)
python query_db.py --query "How to configure BGP?" --n-results 5

# 4. RAG query (requires API key)
export GEMINI_API_KEY='your-key'
python rag_query.py --query "How to configure BGP?" --model gemini
```

## Typical Execution Time

- Chunking: ~1-2 minutes for 139 documents
- Vectorize: ~5-10 minutes (depends on CPU/GPU)
- Query: <1 second

## Common Issues

**Model not found**: Model downloads automatically on first use. Ensure internet connection.

**Out of memory**: Reduce `--batch-size` in vectorize.py (e.g., to 50) or use lighter model: `all-MiniLM-L6-v2`

**Slow processing**: Normal on first run (model download). Subsequent runs are faster.

