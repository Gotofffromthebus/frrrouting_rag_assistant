# Quick Start Guide

## Быстрый старт для создания RAG системы

### 1. Разбиение на чанки

```bash
source venv/bin/activate
python chunking.py --input fr_docs --output chunks.json
```

Это создаст файл `chunks.json` с разбитыми на чанки документами.

### 2. Создание векторной БД

```bash
python vectorize.py --chunks chunks.json --output vector_db
```

Это создаст векторную БД в директории `vector_db/`.

**Примечание:** Первый запуск загрузит модель (около 80MB), это может занять время.

### 3. Тестирование запросов

```bash
# Простой поиск
python query_db.py --query "How to configure BGP?" --n-results 5

# RAG запрос с генерацией ответа
python rag_query.py --query "How to configure BGP?" --model gemini
```

### Полный пример

```bash
# 1. Chunking
python chunking.py --input fr_docs --output chunks.json --min-size 200 --max-size 2000

# 2. Vectorize
python vectorize.py --chunks chunks.json --output vector_db --model all-mpnet-base-v2

# 3. Query
python query_db.py --query "OSPF configuration" --n-results 3

# 4. RAG Query (требует API ключ)
export GEMINI_API_KEY='your-key'
python rag_query.py --query "ip address" --model gemini --min-relevance 0.05
```

## Время выполнения (примерно)

- Chunking: ~1-2 минуты для 139 документов
- Vectorize: ~5-10 минут (зависит от CPU/GPU)
- Query: <1 секунда

## Troubleshooting

**Ошибка: модель не найдена**
- Модель загрузится автоматически при первом использовании
- Убедитесь, что есть интернет-соединение

**Ошибка: Out of memory**
- Уменьшите `--batch-size` в vectorize.py (например, до 50)
- Используйте более легкую модель: `all-MiniLM-L6-v2`

**Медленная обработка**
- Это нормально для первого запуска (загрузка модели)
- Последующие запуски будут быстрее

