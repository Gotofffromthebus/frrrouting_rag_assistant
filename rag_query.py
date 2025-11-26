#!/usr/bin/env python3
"""
RAG система: поиск + генерация ответа через LLM.

Использование:
    python rag_query.py --query "How to configure BGP?" --model openai --api-key YOUR_KEY
    python rag_query.py --query "How to configure BGP?" --model gemini --api-key YOUR_GEMINI_KEY
"""

import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def expand_query_for_commands(query_text: str) -> str:
    """
    Расширяет запрос для лучшего поиска технических команд.
    
    Примеры:
    - "ip address" -> "configure ip address on interface zebra command set"
    - "bgp neighbor" -> "configure bgp neighbor command setup"
    """
    query_lower = query_text.lower()
    
    # Технические термины, которые нужно расширить
    tech_terms = {
        'ip address': 'configure ip address on interface zebra command set',
        'ipv6 address': 'configure ipv6 address on interface zebra command set',
        'bgp': 'border gateway protocol bgp configuration command',
        'ospf': 'open shortest path first ospf configuration command',
        'interface': 'network interface configuration zebra command',
        'neighbor': 'bgp neighbor peer configuration command',
        'route': 'routing table route configuration command',
        'static route': 'static routing configuration command',
    }
    
    expanded = query_text
    for term, expansion in tech_terms.items():
        if term in query_lower:
            expanded = f"{expanded} {expansion}"
            break  # Расширяем только первый найденный термин
    
    return expanded

def search_in_db(db_path: str, collection_name: str, query_text: str, 
                min_relevance: float = 0.1,
                max_results: int = 20,
                model_name: str = "all-mpnet-base-v2",
                auto_adjust: bool = True,
                use_hybrid: bool = True):
    """
    Поиск релевантных чанков в векторной БД с использованием порога релевантности.
    Поддерживает гибридный поиск: семантический + по ключевым словам.
    
    Args:
        db_path: Путь к векторной БД
        collection_name: Название коллекции
        query_text: Текст запроса
        min_relevance: Минимальная релевантность (0.0-1.0). Результаты с меньшей релевантностью отбрасываются.
        max_results: Максимальное количество результатов для проверки
        model_name: Название модели для embeddings
        auto_adjust: Автоматически снижать порог, если результатов нет
        use_hybrid: Использовать гибридный поиск (семантический + ключевые слова)
    
    Returns:
        Результаты поиска с отфильтрованными по порогу релевантности и фактический использованный порог
    """
    # Подключение к БД
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Ошибка: коллекция '{collection_name}' не найдена")
        print(f"Убедитесь, что векторная БД создана: python vectorize.py")
        return None, min_relevance
    
    # Загрузка модели для embeddings
    model = SentenceTransformer(model_name)
    
    # Расширяем запрос для лучшего поиска команд
    expanded_query = expand_query_for_commands(query_text)
    
    # Создание embedding для запроса (используем расширенный запрос)
    query_embedding = model.encode([expanded_query], show_progress_bar=False).tolist()[0]
    
    # Увеличиваем количество результатов для гибридного поиска
    search_n_results = max_results * 2 if use_hybrid else max_results
    
    # Поиск с максимальным количеством результатов
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=search_n_results
    )
    
    # Если включен гибридный поиск, добавляем результаты по ключевым словам
    if use_hybrid and results and 'documents' in results and len(results['documents'][0]) > 0:
        # Извлекаем ключевые слова из запроса (технические термины)
        keywords = [word.lower() for word in query_text.split() if len(word) > 2]
        
        # Ищем дополнительные результаты по ключевым словам через where_document
        semantic_ids = set(results['ids'][0] if 'ids' in results and results['ids'] else [])
        
        # Для каждого ключевого слова ищем документы, содержащие его
        keyword_docs = []
        keyword_metas = []
        keyword_ids = []
        keyword_distances = []
        
        for keyword in keywords[:3]:  # Ограничиваем количество ключевых слов
            try:
                # Поиск документов, содержащих ключевое слово
                keyword_results = collection.query(
                    query_embeddings=[query_embedding],  # Используем тот же embedding
                    n_results=min(5, max_results),
                    where_document={"$contains": keyword}  # Фильтр по содержимому
                )
                
                if keyword_results and 'ids' in keyword_results:
                    for i, kid in enumerate(keyword_results['ids'][0]):
                        if kid not in semantic_ids and kid not in keyword_ids:
                            keyword_ids.append(kid)
                            if 'documents' in keyword_results:
                                keyword_docs.append(keyword_results['documents'][0][i])
                            if 'metadatas' in keyword_results:
                                keyword_metas.append(keyword_results['metadatas'][0][i])
                            if 'distances' in keyword_results:
                                # Для keyword поиска используем немного большее расстояние
                                dist = keyword_results['distances'][0][i] if i < len(keyword_results['distances'][0]) else 0.6
                                keyword_distances.append(dist)
            except Exception:
                # Если where_document не поддерживается, пропускаем
                continue
        
        # Объединяем результаты
        if keyword_docs:
            results['documents'][0].extend(keyword_docs)
            results['metadatas'][0].extend(keyword_metas)
            results['ids'][0].extend(keyword_ids)
            results['distances'][0].extend(keyword_distances)
    
    # Фильтруем по порогу релевантности
    actual_threshold = min_relevance
    
    if 'distances' in results and results['distances'] and len(results['distances'][0]) > 0:
        distances = results['distances'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        
        # Фильтруем результаты по порогу релевантности
        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        filtered_distances = []
        
        for i, distance in enumerate(distances):
            relevance = max(0, 1 - distance)  # Защита от отрицательных значений
            if relevance >= min_relevance:
                filtered_docs.append(documents[i])
                filtered_metas.append(metadatas[i])
                filtered_ids.append(ids[i])
                filtered_distances.append(distance)
        
        # Если ничего не найдено и включена автоматическая адаптация
        if len(filtered_docs) == 0 and auto_adjust and min_relevance > 0.05:
            # Находим лучшую релевантность среди всех результатов
            best_relevance = max([max(0, 1 - d) for d in distances]) if distances else 0
            
            # Если есть результаты с релевантностью > 0.05, используем их
            if best_relevance >= 0.05:
                # Используем порог чуть ниже лучшей релевантности, но не ниже 0.05
                actual_threshold = max(0.05, best_relevance - 0.05)
                
                # Перефильтровываем с новым порогом
                filtered_docs = []
                filtered_metas = []
                filtered_ids = []
                filtered_distances = []
                
                for i, distance in enumerate(distances):
                    relevance = max(0, 1 - distance)
                    if relevance >= actual_threshold:
                        filtered_docs.append(documents[i])
                        filtered_metas.append(metadatas[i])
                        filtered_ids.append(ids[i])
                        filtered_distances.append(distance)
        
        # Обновляем результаты
        results['documents'] = [filtered_docs]
        results['metadatas'] = [filtered_metas]
        results['ids'] = [filtered_ids]
        results['distances'] = [filtered_distances]
        results['actual_threshold'] = actual_threshold  # Сохраняем фактический порог
    
    return results, actual_threshold

def format_context(results, show_relevance: bool = False):
    """
    Форматирует найденные чанки в контекст для LLM.
    
    Args:
        results: Результаты поиска из ChromaDB
        show_relevance: Показывать ли релевантность в контексте (для отладки)
    """
    context_parts = []
    
    distances = results.get('distances', [[]])[0] if 'distances' in results else []
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        title = metadata.get('title', 'Unknown')
        section = metadata.get('section', 'N/A')
        url = metadata.get('url', 'N/A')
        
        relevance_info = ""
        if show_relevance and i <= len(distances):
            relevance = max(0, 1 - distances[i-1])
            relevance_info = f" [Релевантность: {relevance:.3f}]"
        
        context_parts.append(f"""
[Документ {i}]{relevance_info}
Заголовок: {title}
Раздел: {section}
URL: {url}
Содержание:
{doc}
""")
    
    return "\n".join(context_parts)

def generate_answer_openai(query: str, context: str, api_key: str, model: str = "gpt-4o-mini"):
    """
    Генерирует ответ через OpenAI API.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Ошибка: openai не установлен. Установите: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Ты - помощник по документации FRRouting. Ответь на вопрос пользователя на основе предоставленной документации.

Вопрос пользователя: {query}

Документация:
{context}

Инструкции:
- Ответь четко и по делу
- Используй только информацию из предоставленной документации
- Если пользователь спрашивает о команде конфигурации, ОБЯЗАТЕЛЬНО приведи точную команду из документации
- Для команд используй формат: **Команда:** `команда`
- Приведи конкретные примеры команд, если они есть в документации
- Если информации недостаточно, скажи об этом
- Укажи источники (URL) в конце ответа

ВАЖНО: Если вопрос о настройке или конфигурации, обязательно покажи точную команду из документации!

Ответ:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты - эксперт по FRRouting, помогающий пользователям с конфигурацией и настройкой."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Ошибка при обращении к OpenAI API: {e}")
        return None

def generate_answer_gemini(query: str, context: str, api_key: str, model: str = "gemini-2.5-flash"):
    """
    Генерирует ответ через Google Gemini API.
    
    Args:
        query: Вопрос пользователя
        context: Контекст из найденных чанков
        api_key: Google Gemini API ключ
        model: Название модели (gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro)
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Ошибка: google-generativeai не установлен. Установите: pip install google-generativeai")
        return None
    
    # Настраиваем API
    genai.configure(api_key=api_key)
    
    # Проверяем доступные модели и нормализуем название
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Модели приходят с префиксом "models/", но GenerativeModel принимает без префикса
        available_short = [m.replace("models/", "") for m in available_models]
        
        # Нормализуем название модели (убираем префикс если есть)
        model_short = model.replace("models/", "")
        
        if model_short not in available_short:
            # Автоматический выбор доступной модели
            # Сначала ищем flash модели (быстрее)
            flash_models = [m for m in available_short if "flash" in m.lower()]
            pro_models = [m for m in available_short if "pro" in m.lower()]
            
            selected_model = None
            
            # Приоритет: flash модели (быстрее)
            if flash_models:
                # Предпочитаем 2.5, потом 1.5
                for preferred in ["gemini-2.5-flash", "gemini-1.5-flash"]:
                    for flash in flash_models:
                        if flash == preferred or flash.startswith(preferred):
                            selected_model = flash
                            break
                    if selected_model:
                        break
                
                # Если не нашли предпочитаемую, берем первую flash
                if not selected_model:
                    selected_model = flash_models[0]
            
            # Если flash нет, берем pro
            elif pro_models:
                for preferred in ["gemini-2.5-pro", "gemini-1.5-pro"]:
                    for pro in pro_models:
                        if pro == preferred or pro.startswith(preferred):
                            selected_model = pro
                            break
                    if selected_model:
                        break
                
                if not selected_model:
                    selected_model = pro_models[0]
            
            if selected_model:
                print(f"⚠️  Модель '{model}' не найдена, используется '{selected_model}'")
                model = selected_model
            else:
                print(f"⚠️  Модель '{model}' не найдена, используется первая доступная")
                model = available_short[0] if available_short else model_short
        else:
            model = model_short
    except Exception as e:
        print(f"⚠️  Не удалось получить список моделей: {e}")
        # Продолжаем с указанной моделью (убираем префикс если есть)
        model = model.replace("models/", "")
    
    prompt = f"""Ты - помощник по документации FRRouting. Ответь на вопрос пользователя на основе предоставленной документации.

Вопрос пользователя: {query}

Документация:
{context}

Инструкции:
- Ответь четко и по делу
- Используй только информацию из предоставленной документации
- Если пользователь спрашивает о команде конфигурации, ОБЯЗАТЕЛЬНО приведи точную команду из документации
- Для команд используй формат: **Команда:** `команда`
- Приведи конкретные примеры команд, если они есть в документации
- Если информации недостаточно, скажи об этом
- Укажи источники (URL) в конце ответа

ВАЖНО: Если вопрос о настройке или конфигурации, обязательно покажи точную команду из документации!

Ответ:"""
    
    try:
        # Создаем модель (название уже нормализовано)
        model_instance = genai.GenerativeModel(model)
        
        # Генерируем ответ
        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 2000,  # Увеличено для более длинных ответов
        }
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Проверяем результат
        if not response.candidates:
            return "Ошибка: модель не вернула ответ"
        
        candidate = response.candidates[0]
        
        # Получаем finish_reason (может быть enum или число)
        finish_reason = candidate.finish_reason
        finish_reason_str = str(finish_reason)
        finish_reason_name = finish_reason.name if hasattr(finish_reason, 'name') else finish_reason_str
        
        # Пытаемся получить текст ответа
        try:
            answer_text = response.text.strip()
            # Если текст получен, возвращаем его (даже если finish_reason не STOP)
            if answer_text:
                return answer_text
        except (ValueError, AttributeError) as e:
            # Если не удалось получить текст, обрабатываем finish_reason
            pass
        
        # Обрабатываем разные finish_reason когда текст недоступен
        if "SAFETY" in finish_reason_name.upper() or finish_reason == 3:
            return "Извините, ответ был заблокирован системой безопасности. Попробуйте переформулировать запрос."
        elif "MAX_TOKENS" in finish_reason_name.upper() or finish_reason == 2:
            return "Ответ обрезан из-за лимита токенов. Попробуйте уменьшить размер контекста или увеличить max_output_tokens."
        elif "RECITATION" in finish_reason_name.upper() or finish_reason == 4:
            return "Ответ может содержать скопированный контент. Попробуйте переформулировать запрос."
        else:
            return f"Ошибка: finish_reason={finish_reason_name} ({finish_reason}). Попробуйте переформулировать запрос."
        
    except Exception as e:
        print(f"❌ Ошибка при обращении к Gemini API: {e}")
        print(f"   Попробуйте другую модель: --llm-model gemini-2.5-flash")
        return None

def main():
    parser = argparse.ArgumentParser(description="RAG система: поиск + генерация ответа")
    parser.add_argument("--query", "-q", required=True, help="Вопрос пользователя")
    parser.add_argument("--db", default="vector_db", help="Путь к векторной БД")
    parser.add_argument("--collection", default="frr_docs", help="Название коллекции")
    parser.add_argument("--min-relevance", type=float, default=0.1,
                       help="Минимальная релевантность результатов (0.0-1.0). По умолчанию: 0.1")
    parser.add_argument("--max-results", type=int, default=20,
                       help="Максимальное количество результатов для проверки. По умолчанию: 20")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="Модель для embeddings")
    parser.add_argument("--show-relevance", action="store_true",
                       help="Показать релевантность найденных фрагментов")
    parser.add_argument("--no-auto-adjust", action="store_true",
                       help="Отключить автоматическую адаптацию порога релевантности")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Отключить гибридный поиск (только семантический)")
    
    # LLM выбор
    parser.add_argument("--model", choices=["openai", "gemini", "local"], default="gemini", 
                       help="Провайдер для генерации ответа (openai, gemini, local)")
    parser.add_argument("--api-key", help="API ключ (OpenAI или Gemini, или установите переменную окружения)")
    parser.add_argument("--llm-model", default="gemini-2.5-flash", 
                       help="Модель LLM (для OpenAI: gpt-4o-mini, gpt-4; для Gemini: gemini-2.5-flash, gemini-2.5-pro)")
    parser.add_argument("--show-sources", action="store_true", 
                       help="Показать источники (URL найденных документов)")
    
    args = parser.parse_args()
    
    # Поиск в векторной БД
    print(f"🔍 Поиск релевантной информации для: '{args.query}'")
    print(f"   Порог релевантности: {args.min_relevance}, Максимум результатов для проверки: {args.max_results}")
    
    results, actual_threshold = search_in_db(
        db_path=args.db,
        collection_name=args.collection,
        query_text=args.query,
        min_relevance=args.min_relevance,
        max_results=args.max_results,
        model_name=args.embedding_model,
        auto_adjust=not args.no_auto_adjust,
        use_hybrid=not args.no_hybrid
    )
    
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        print("❌ Релевантная информация не найдена")
        print(f"   Попробуйте уменьшить --min-relevance (текущее значение: {args.min_relevance})")
        return
    
    # Показываем, если порог был автоматически скорректирован
    if actual_threshold < args.min_relevance:
        print(f"   ⚠️  Порог автоматически снижен до {actual_threshold:.3f} (было {args.min_relevance})")
    
    num_found = len(results['documents'][0])
    print(f"✅ Найдено {num_found} релевантных фрагментов (релевантность >= {actual_threshold:.3f})")
    
    # Показываем релевантность если нужно
    if args.show_relevance and 'distances' in results and results['distances']:
        print("\n📊 Релевантность найденных фрагментов:")
        for i, distance in enumerate(results['distances'][0][:5], 1):  # Показываем топ-5
            relevance = max(0, 1 - distance)
            title = results['metadatas'][0][i-1].get('title', 'Unknown')[:50]
            print(f"   {i}. {relevance:.3f} - {title}...")
        if num_found > 5:
            print(f"   ... и еще {num_found - 5} фрагментов")
        print()
    
    # Форматируем контекст
    context = format_context(results, show_relevance=args.show_relevance)
    
    # Генерируем ответ
    print("🤖 Генерация ответа...\n")
    
    if args.model == "openai":
        # Получаем API ключ
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ Ошибка: OpenAI API ключ не указан")
            print("   Используйте --api-key или установите переменную OPENAI_API_KEY")
            return
        
        answer = generate_answer_openai(
            query=args.query,
            context=context,
            api_key=api_key,
            model=args.llm_model
        )
    elif args.model == "gemini":
        # Получаем API ключ
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ Ошибка: Gemini API ключ не указан")
            print("   Используйте --api-key или установите переменную GEMINI_API_KEY")
            print("   Получить ключ: https://makersuite.google.com/app/apikey")
            return
        
        answer = generate_answer_gemini(
            query=args.query,
            context=context,
            api_key=api_key,
            model=args.llm_model
        )
    elif args.model == "local":
        print("Локальные LLM пока не реализованы.")
        return
    
    if answer:
        print("=" * 80)
        print("📝 ОТВЕТ:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # Показываем источники если нужно
        if args.show_sources:
            print("\n📚 Источники:")
            for i, metadata in enumerate(results['metadatas'][0], 1):
                url = metadata.get('url', 'N/A')
                title = metadata.get('title', 'Unknown')
                print(f"  {i}. {title}")
                print(f"     {url}")
    else:
        print("❌ Не удалось сгенерировать ответ")

if __name__ == "__main__":
    main()

