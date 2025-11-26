#!/usr/bin/env python3
"""
Скрипт для создания векторной БД из чанков.

Использование:
    python vectorize.py --chunks chunks.json --output vector_db --model all-mpnet-base-v2
"""

import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from tqdm import tqdm

def create_vector_db(chunks_file: str, output_dir: str, model_name: str = "all-mpnet-base-v2",
                   collection_name: str = "frr_docs", batch_size: int = 100):
    """
    Создает векторную БД из чанков.
    
    Args:
        chunks_file: JSON файл с чанками
        output_dir: Директория для сохранения БД
        model_name: Название модели для embeddings
        collection_name: Название коллекции
        batch_size: Размер батча для обработки
    """
    # Загружаем чанки
    print(f"Загрузка чанков из {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Загружено {len(chunks)} чанков")
    
    # Загружаем модель для embeddings
    print(f"Загрузка модели: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Инициализируем Chroma
    print(f"Инициализация Chroma DB в {output_dir}...")
    client = chromadb.PersistentClient(
        path=output_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Создаем или получаем коллекцию
    try:
        # Пытаемся получить существующую коллекцию
        existing_collection = client.get_collection(name=collection_name)
        print(f"Найдена существующая коллекция: {collection_name}")
        # Удаляем её полностью
        client.delete_collection(name=collection_name)
        print("Коллекция удалена")
    except Exception:
        # Коллекция не существует, это нормально
        pass

    # Создаем новую коллекцию
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "FRRouting documentation chunks"}
    )
    print(f"Создана коллекция: {collection_name}")
    
    # Обрабатываем чанки батчами
    print("Создание embeddings и загрузка в БД...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(chunks), batch_size), desc="Обработка батчей"):
        batch = chunks[batch_idx:batch_idx + batch_size]
        
        # Извлекаем тексты
        texts = [chunk['text'] for chunk in batch]
        
        # Создаем embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        # Подготавливаем данные для Chroma
        ids = [f"chunk_{batch_idx + i}" for i in range(len(batch))]
        documents = texts
        metadatas = []
        
        for chunk in batch:
            # Chroma требует, чтобы метаданные были простыми типами
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)
        
        # Добавляем в коллекцию
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    print(f"\n✅ Векторная БД создана: {output_dir}")
    print(f"   Коллекция: {collection_name}")
    print(f"   Чанков: {len(chunks)}")
    
    return collection

def query_example(collection, query_text: str, n_results: int = 5):
    """Пример запроса к векторной БД."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\nРезультаты для запроса: '{query_text}'")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n[{i}] {metadata.get('title', 'Unknown')}")
        print(f"    Раздел: {metadata.get('section', 'N/A')}")
        print(f"    {doc[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="Создание векторной БД из чанков")
    parser.add_argument("--chunks", "-c", default="chunks.json", help="JSON файл с чанками")
    parser.add_argument("--output", "-o", default="vector_db", help="Директория для векторной БД")
    parser.add_argument("--model", "-m", default="all-mpnet-base-v2", 
                       help="Модель для embeddings (sentence-transformers)")
    parser.add_argument("--collection", default="frr_docs", help="Название коллекции")
    parser.add_argument("--batch-size", type=int, default=100, help="Размер батча")
    parser.add_argument("--test-query", help="Тестовый запрос после создания БД")
    args = parser.parse_args()
    
    # Создаем векторную БД
    collection = create_vector_db(
        chunks_file=args.chunks,
        output_dir=args.output,
        model_name=args.model,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    # Тестовый запрос если указан
    if args.test_query:
        query_example(collection, args.test_query)

if __name__ == "__main__":
    main()

