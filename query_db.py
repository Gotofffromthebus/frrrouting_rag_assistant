#!/usr/bin/env python3
"""Query vector database."""

import argparse
import chromadb
from sentence_transformers import SentenceTransformer

def query_vector_db(db_path: str, collection_name: str, query_text: str, 
                   n_results: int = 5, model_name: str = "all-mpnet-base-v2",
                   filter_dict: dict = None):
    """Query vector database."""
    print(f"Connecting to DB: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' found ({collection.count()} chunks)")
    except Exception as e:
        print(f"Error: collection '{collection_name}' not found")
        print(f"Create vector DB first: python vectorize.py")
        return
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Processing query: '{query_text}'")
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]
    
    print(f"Searching {n_results} most relevant results...")
    
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    if filter_dict:
        query_kwargs["where"] = filter_dict
        print(f"Applied filter: {filter_dict}")
    
    results = collection.query(**query_kwargs)
    
    print("\n" + "=" * 80)
    print(f"Search results for: '{query_text}'")
    print("=" * 80)
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found")
        return
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0] if 'distances' in results else [None] * len(results['documents'][0])
    ), 1):
        if distance is not None:
            relevance = max(0, 1 - distance)
            print(f"\n[{i}] Relevance: {relevance:.4f}")
        else:
            print(f"\n[{i}]")
        print(f"Title: {metadata.get('title', 'Unknown')}")
        
        if 'section' in metadata:
            print(f"Section: {metadata['section']}")
        
        if 'url' in metadata:
            print(f"URL: {metadata['url']}")
        
        print(f"\nText (first 300 chars):")
        print("-" * 80)
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Query vector database")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--db", default="vector_db", help="Path to vector DB")
    parser.add_argument("--collection", default="frr_docs", help="Collection name")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    args = parser.parse_args()
    
    query_vector_db(
        db_path=args.db,
        collection_name=args.collection,
        query_text=args.query,
        n_results=args.n_results,
        model_name=args.model
    )

if __name__ == "__main__":
    main()

