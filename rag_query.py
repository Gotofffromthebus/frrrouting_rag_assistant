#!/usr/bin/env python3
"""RAG system: search + LLM answer generation."""

import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def expand_query_for_commands(query_text: str) -> str:
    """Expand query for better technical command search."""
    query_lower = query_text.lower()
    
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
            break
    
    return expanded

def search_in_db(db_path: str, collection_name: str, query_text: str, 
                min_relevance: float = 0.1,
                max_results: int = 20,
                model_name: str = "all-mpnet-base-v2",
                auto_adjust: bool = True,
                use_hybrid: bool = True):
    """Search relevant chunks in vector DB with relevance threshold. Supports hybrid search."""
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: collection '{collection_name}' not found")
        print(f"Create vector DB first: python vectorize.py")
        return None, min_relevance
    
    model = SentenceTransformer(model_name)
    expanded_query = expand_query_for_commands(query_text)
    query_embedding = model.encode([expanded_query], show_progress_bar=False).tolist()[0]
    
    search_n_results = max_results * 2 if use_hybrid else max_results
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=search_n_results
    )
    
    if use_hybrid and results and 'documents' in results and len(results['documents'][0]) > 0:
        keywords = [word.lower() for word in query_text.split() if len(word) > 2]
        semantic_ids = set(results['ids'][0] if 'ids' in results and results['ids'] else [])
        
        keyword_docs = []
        keyword_metas = []
        keyword_ids = []
        keyword_distances = []
        
        for keyword in keywords[:3]:
            try:
                keyword_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(5, max_results),
                    where_document={"$contains": keyword}
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
                                dist = keyword_results['distances'][0][i] if i < len(keyword_results['distances'][0]) else 0.6
                                keyword_distances.append(dist)
            except Exception:
                continue
        
        if keyword_docs:
            results['documents'][0].extend(keyword_docs)
            results['metadatas'][0].extend(keyword_metas)
            results['ids'][0].extend(keyword_ids)
            results['distances'][0].extend(keyword_distances)
    
    actual_threshold = min_relevance
    
    if 'distances' in results and results['distances'] and len(results['distances'][0]) > 0:
        distances = results['distances'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        
        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        filtered_distances = []
        
        for i, distance in enumerate(distances):
            relevance = max(0, 1 - distance)
            if relevance >= min_relevance:
                filtered_docs.append(documents[i])
                filtered_metas.append(metadatas[i])
                filtered_ids.append(ids[i])
                filtered_distances.append(distance)
        
        if len(filtered_docs) == 0 and auto_adjust and min_relevance > 0.05:
            best_relevance = max([max(0, 1 - d) for d in distances]) if distances else 0
            
            if best_relevance >= 0.05:
                actual_threshold = max(0.05, best_relevance - 0.05)
                
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
        
        results['documents'] = [filtered_docs]
        results['metadatas'] = [filtered_metas]
        results['ids'] = [filtered_ids]
        results['distances'] = [filtered_distances]
        results['actual_threshold'] = actual_threshold
    
    return results, actual_threshold

def format_context(results, show_relevance: bool = False, max_chunks: int = 15):
    """Format found chunks into context for LLM."""
    context_parts = []
    
    distances = results.get('distances', [[]])[0] if 'distances' in results else []
    documents = results['documents'][0][:max_chunks]
    metadatas = results['metadatas'][0][:max_chunks]
    
    for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
        title = metadata.get('title', 'Unknown')
        section = metadata.get('section', 'N/A')
        url = metadata.get('url', 'N/A')
        
        relevance_info = ""
        if show_relevance and i <= len(distances):
            relevance = max(0, 1 - distances[i-1])
            relevance_info = f" [Relevance: {relevance:.3f}]"
        
        context_parts.append(f"""
[Document {i}]{relevance_info}
Title: {title}
Section: {section}
URL: {url}
Content:
{doc}
""")
    
    return "\n".join(context_parts)

def generate_answer_openai(query: str, context: str, api_key: str, model: str = "gpt-4o-mini"):
    """Generate answer via OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai not installed. Install: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Ты - эксперт по FRRouting, помогающий пользователям с конфигурацией.

Вопрос пользователя: {query}

Документация:
{context}

Формат ответа:
1. **Краткое введение** (1-2 предложения) - что нужно сделать и зачем
2. **Пошаговая инструкция** - четкие шаги с командами в правильном порядке
3. **Пример конфигурации** - полный рабочий пример с реальными значениями
4. **Дополнительные команды** (если есть) - опциональные настройки

Правила:
- Используй ТОЛЬКО информацию из предоставленной документации
- Для команд используй формат: `команда` (в обратных кавычках)
- Структурируй ответ как практическую инструкцию для администратора сети
- Если вопрос "как настроить/сконфигурировать", дай пошаговую инструкцию
- Начинай каждый шаг с действия (например: "1. Войдите в режим конфигурации интерфейса")
- Показывай примеры с реальными значениями из документации
- Укажи источники (URL) в конце

ВАЖНО: Ответ должен быть практичным и готовым к использованию!

Ответ:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты - эксперт по FRRouting. Твоя задача - давать четкие, пошаговые инструкции по конфигурации. Структурируй ответы как практические руководства для администраторов сетей."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def generate_answer_gemini(query: str, context: str, api_key: str, model: str = "gemini-2.5-flash"):
    """Generate answer via Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: google-generativeai not installed. Install: pip install google-generativeai")
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
                print(f"Warning: model '{model}' not found, using '{selected_model}'")
                model = selected_model
            else:
                print(f"Warning: model '{model}' not found, using first available")
                model = available_short[0] if available_short else model_short
        else:
            model = model_short
    except Exception as e:
        print(f"Warning: failed to get model list: {e}")
        # Продолжаем с указанной моделью (убираем префикс если есть)
        model = model.replace("models/", "")
    
    prompt = f"""Ты - эксперт по FRRouting, помогающий пользователям с конфигурацией.

Вопрос пользователя: {query}

Документация:
{context}

Формат ответа:
1. **Краткое введение** (1-2 предложения) - что нужно сделать и зачем
2. **Пошаговая инструкция** - четкие шаги с командами в правильном порядке
3. **Пример конфигурации** - полный рабочий пример с реальными значениями
4. **Дополнительные команды** (если есть) - опциональные настройки

Правила:
- Используй ТОЛЬКО информацию из предоставленной документации
- Для команд используй формат: `команда` (в обратных кавычках)
- Структурируй ответ как практическую инструкцию для администратора сети
- Если вопрос "как настроить/сконфигурировать", дай пошаговую инструкцию
- Начинай каждый шаг с действия (например: "1. Войдите в режим конфигурации интерфейса")
- Показывай примеры с реальными значениями из документации
- Укажи источники (URL) в конце

ВАЖНО: Ответ должен быть практичным и готовым к использованию!

Ответ:"""
    
    try:
        # Создаем модель (название уже нормализовано)
        model_instance = genai.GenerativeModel(model)
        
        # Генерируем ответ
        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 8000,  # Увеличено для более длинных ответов
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
        print(f"Error calling Gemini API: {e}")
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
    print(f"Searching for: '{args.query}'")
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
        print("No relevant information found")
        print(f"   Try lowering --min-relevance (current: {args.min_relevance})")
        return
    
    # Показываем, если порог был автоматически скорректирован
    if actual_threshold < args.min_relevance:
        print(f"   Threshold automatically lowered to {actual_threshold:.3f} (was {args.min_relevance})")
    
    num_found = len(results['documents'][0])
    print(f"Found {num_found} relevant fragments (relevance >= {actual_threshold:.3f})")
    
    # Показываем релевантность если нужно
    if args.show_relevance and 'distances' in results and results['distances']:
        print("\nRelevance of found fragments:")
        for i, distance in enumerate(results['distances'][0][:5], 1):  # Показываем топ-5
            relevance = max(0, 1 - distance)
            title = results['metadatas'][0][i-1].get('title', 'Unknown')[:50]
            print(f"   {i}. {relevance:.3f} - {title}...")
        if num_found > 5:
            print(f"   ... and {num_found - 5} more fragments")
        print()
    
    # Форматируем контекст (ограничиваем количество чанков для LLM)
    max_context_chunks = min(15, num_found)  # Используем максимум 15 самых релевантных чанков
    if num_found > max_context_chunks:
        print(f"   Using top {max_context_chunks} most relevant chunks for LLM context")
    context = format_context(results, show_relevance=args.show_relevance, max_chunks=max_context_chunks)
    
    # Генерируем ответ
    print("Generating answer...\n")
    
    if args.model == "openai":
        # Получаем API ключ
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not specified")
            print("   Use --api-key or set OPENAI_API_KEY env var")
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
            print("Error: Gemini API key not specified")
            print("   Use --api-key or set GEMINI_API_KEY env var")
            print("   Get key: https://makersuite.google.com/app/apikey")
            return
        
        answer = generate_answer_gemini(
            query=args.query,
            context=context,
            api_key=api_key,
            model=args.llm_model
        )
    elif args.model == "local":
        print("Local LLMs not implemented yet.")
        return
    
    if answer:
        print("=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # Показываем источники если нужно
        if args.show_sources:
            print("\nSources:")
            for i, metadata in enumerate(results['metadatas'][0], 1):
                url = metadata.get('url', 'N/A')
                title = metadata.get('title', 'Unknown')
                print(f"  {i}. {title}")
                print(f"     {url}")
    else:
        print("Failed to generate answer")

if __name__ == "__main__":
    main()

