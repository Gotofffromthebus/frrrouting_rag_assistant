#!/usr/bin/env python3
"""
Скрипт для запросов к векторной БД.

Использование:
    python query_db.py --query "How to configure BGP?" --n-results 5
"""

import argparse
import chromadb
from sentence_transformers import SentenceTransformer

def query_vector_db(db_path: str, collection_name: str, query_text: str, 
                   n_results: int = 5, model_name: str = "all-mpnet-base-v2",
                   filter_dict: dict = None):
    """
    Выполняет запрос к векторной БД.
    
    Args:
        db_path: Путь к векторной БД
        collection_name: Название коллекции
        query_text: Текст запроса
        n_results: Количество результатов
        model_name: Название модели для embeddings
        filter_dict: Словарь для фильтрации по метаданным
    """
    # Подключение к БД
    print(f"Подключение к БД: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Коллекция '{collection_name}' найдена ({collection.count()} чанков)")
    except Exception as e:
        print(f"Ошибка: коллекция '{collection_name}' не найдена")
        print(f"Убедитесь, что векторная БД создана: python vectorize.py")
        return
    
    # Загрузка модели
    print(f"Загрузка модели: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Создание embedding для запроса
    print(f"Обработка запроса: '{query_text}'")
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]
    
    # Поиск
    print(f"Поиск {n_results} наиболее релевантных результатов...")
    
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    if filter_dict:
        query_kwargs["where"] = filter_dict
        print(f"Применен фильтр: {filter_dict}")
    
    results = collection.query(**query_kwargs)
    
    # Вывод результатов
    print("\n" + "=" * 80)
    print(f"Результаты поиска для: '{query_text}'")
    print("=" * 80)
    
    if not results['documents'] or not results['documents'][0]:
        print("Результаты не найдены")
        return
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0] if 'distances' in results else [None] * len(results['documents'][0])
    ), 1):
        # Исправляем расчет релевантности (distance может быть > 1)
        if distance is not None:
            relevance = max(0, 1 - distance)  # Защита от отрицательных значений
            print(f"\n[{i}] Релевантность: {relevance:.4f}")
        else:
            print(f"\n[{i}]")
        print(f"Заголовок: {metadata.get('title', 'Unknown')}")
        
        if 'section' in metadata:
            print(f"Раздел: {metadata['section']}")
        
        if 'url' in metadata:
            print(f"URL: {metadata['url']}")
        
        print(f"\nТекст (первые 300 символов):")
        print("-" * 80)
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Запросы к векторной БД")
    parser.add_argument("--query", "-q", required=True, help="Текст запроса")
    parser.add_argument("--db", default="vector_db", help="Путь к векторной БД")
    parser.add_argument("--collection", default="frr_docs", help="Название коллекции")
    parser.add_argument("--n-results", type=int, default=5, help="Количество результатов")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Модель для embeddings")
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

