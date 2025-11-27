# RAG Pipeline для FRRouting Documentation

Этот проект содержит инструменты для создания RAG (Retrieval-Augmented Generation) системы на основе документации FRRouting.

## Структура проекта

```
fr-docs-scraper/
├── scraper.py          # Скрапер документации (Markdown + JSON метаданные)
├── chunking.py         # Разбиение документов на чанки
├── vectorize.py        # Создание векторной БД
├── query_db.py         # Простой поиск в векторной БД
├── rag_query.py        # RAG система: поиск + генерация ответа
├── test_gemini.py      # Тест доступности Gemini API
├── fr_docs/            # Спарсенные документы (.md + .metadata.json)
├── chunks.json         # Чанки документов (создается chunking.py)
└── vector_db/          # Векторная БД Chroma (создается vectorize.py)
```

## Установка зависимостей

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

## Процесс работы

### 1. Скрапинг документации (уже выполнено)

```bash
python scraper.py \
  --base-url https://docs.frrouting.org/en/latest/ \
  --output fr_docs \
  --delay 0.1 \
  --max-depth 6 \
  --log-file scraper.log
```

Результат: Markdown файлы в `fr_docs/` с метаданными в JSON.

### 2. Разбиение на чанки

```bash
python chunking.py \
  --input fr_docs \
  --output chunks.json \
  --min-size 200 \
  --max-size 2000 \
  --overlap 100
```

**Стратегия chunking:**
- **Разбиение по заголовкам Markdown** (#, ##, ###, и т.д.) - основной принцип
- Каждый заголовок начинает новый семантический чанк
- Сохранение иерархии разделов в метаданных
- Если чанк превышает `--max-size`, он дополнительно разбивается по параграфам
- Перекрытие между чанками (`--overlap`) для сохранения контекста

**Параметры:**
- `--input`: Директория с Markdown файлами
- `--output`: JSON файл для сохранения чанков
- `--min-size`: Минимальный размер чанка (символов) - фильтрует слишком маленькие чанки
- `--max-size`: Максимальный размер чанка (символов) - если чанк больше, разбивается по параграфам
- `--overlap`: Перекрытие между чанками для контекста (символов)

### 3. Создание векторной БД

```bash
python vectorize.py \
  --chunks chunks.json \
  --output vector_db \
  --model all-mpnet-base-v2 \
  --collection frr_docs \
  --batch-size 100
```

**Параметры:**
- `--chunks`: JSON файл с чанками
- `--output`: Директория для векторной БД (Chroma)
- `--model`: Модель для embeddings (sentence-transformers)
- `--collection`: Название коллекции в БД
- `--batch-size`: Размер батча для обработки

**Доступные модели:**
- `all-MiniLM-L6-v2` - быстрая, английская (384 dim)
- `all-mpnet-base-v2` - более точная, английская (768 dim) ⭐ Рекомендуется
- `paraphrase-multilingual-MiniLM-L12-v2` - мультиязычная (384 dim)

### 4. Использование RAG системы

```bash
# Установите API ключ
export GEMINI_API_KEY='your-key'

# Базовый запрос
python rag_query.py --query "How to configure BGP?" --model gemini

# С настройками поиска
python rag_query.py \
  --query "ip address" \
  --model gemini \
  --min-relevance 0.05 \
  --max-results 20 \
  --show-relevance
```

**Параметры поиска:**
- `--min-relevance`: Минимальная релевантность (0.0-1.0). По умолчанию: 0.1
- `--max-results`: Максимальное количество результатов для проверки. По умолчанию: 20
- `--show-relevance`: Показать релевантность найденных фрагментов
- `--no-auto-adjust`: Отключить автоматическую адаптацию порога
- `--no-hybrid`: Отключить гибридный поиск (только семантический)

**Особенности:**
- ✅ Автоматическая адаптация порога релевантности
- ✅ Гибридный поиск (семантический + ключевые слова)
- ✅ Query expansion для технических терминов
- ✅ Улучшенные промпты для возврата конкретных команд

## Формат данных

### Chunks JSON

```json
{
  "text": "Содержимое чанка...",
  "metadata": {
    "url": "https://docs.frrouting.org/...",
    "title": "BGP",
    "section": "Protocols > BGP > Configuration",
    "section_title": "Configuration",
    "chunk_index": 0,
    "chunk_size": 1234,
    "scraped_at": "2025-11-25T...",
    "source": "fr-docs-scraper"
  }
}
```

## Рекомендации

### Размеры чанков
- **Минимальный**: 200-300 символов (слишком маленькие теряют контекст)
- **Максимальный**: 1500-2000 символов (зависит от модели LLM)
- **Перекрытие**: 50-100 символов (сохраняет контекст между чанками)

### Модели embeddings
- Для английской документации: `all-mpnet-base-v2` ⭐ (лучшее качество)
- Для быстрого поиска: `all-MiniLM-L6-v2`
- Для мультиязычной: `paraphrase-multilingual-MiniLM-L12-v2`

### Пороги релевантности
- **0.3+**: Очень релевантные результаты
- **0.1-0.3**: Хорошие результаты
- **0.05-0.1**: Приемлемые результаты (для общих запросов)
- **<0.05**: Низкая релевантность

## Troubleshooting

**Ошибка: sentence-transformers не найден**
```bash
pip install sentence-transformers
```

**Ошибка: chromadb не найден**
```bash
pip install chromadb
```

**Большой размер БД**
- Уменьшите `--max-size` в chunking.py
- Используйте более компактную модель (all-MiniLM-L6-v2)

**Медленная обработка**
- Увеличьте `--batch-size` в vectorize.py
- Используйте GPU для embeddings (если доступно)

**Низкая релевантность результатов**
- Используйте `--min-relevance 0.05` для более мягкого поиска
- Включите гибридный поиск (по умолчанию включен)
- Проверьте качество chunking (разбиение по заголовкам должно сохранять семантику)

