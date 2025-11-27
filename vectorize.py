#!/usr/bin/env python3
"""Create vector database from chunks."""

import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from tqdm import tqdm

def create_vector_db(chunks_file: str, output_dir: str, model_name: str = "all-mpnet-base-v2",
                   collection_name: str = "frr_docs", batch_size: int = 100):
    """Create vector database from chunks."""
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Initializing Chroma DB in {output_dir}...")
    client = chromadb.PersistentClient(
        path=output_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        existing_collection = client.get_collection(name=collection_name)
        print(f"Found existing collection: {collection_name}")
        client.delete_collection(name=collection_name)
        print("Collection deleted")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "FRRouting documentation chunks"}
    )
    print(f"Created collection: {collection_name}")
    
    print("Creating embeddings and loading into DB...")
    
    for batch_idx in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[batch_idx:batch_idx + batch_size]
        texts = [chunk['text'] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        ids = [f"chunk_{batch_idx + i}" for i in range(len(batch))]
        documents = texts
        metadatas = []
        
        for chunk in batch:
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    print(f"\nVector DB created: {output_dir}")
    print(f"Collection: {collection_name}")
    print(f"Chunks: {len(chunks)}")
    
    return collection

def query_example(collection, query_text: str, n_results: int = 5):
    """Example query to vector DB."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\nResults for query: '{query_text}'")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n[{i}] {metadata.get('title', 'Unknown')}")
        print(f"    Section: {metadata.get('section', 'N/A')}")
        print(f"    {doc[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="Create vector database from chunks")
    parser.add_argument("--chunks", "-c", default="chunks.json", help="JSON file with chunks")
    parser.add_argument("--output", "-o", default="vector_db", help="Output directory for vector DB")
    parser.add_argument("--model", "-m", default="all-mpnet-base-v2", 
                       help="Embedding model (sentence-transformers)")
    parser.add_argument("--collection", default="frr_docs", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--test-query", help="Test query after DB creation")
    args = parser.parse_args()
    
    collection = create_vector_db(
        chunks_file=args.chunks,
        output_dir=args.output,
        model_name=args.model,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    if args.test_query:
        query_example(collection, args.test_query)

if __name__ == "__main__":
    main()

