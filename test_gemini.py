#!/usr/bin/env python3
"""
Тестовый скрипт для проверки доступности Gemini API и списка моделей.
"""

import os
import sys

def test_gemini_api(api_key: str = None):
    """
    Проверяет доступность Gemini API и показывает список доступных моделей.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ Ошибка: Gemini API ключ не указан")
            print("   Используйте: export GEMINI_API_KEY='your-key'")
            print("   Или передайте через параметр: python test_gemini.py --api-key YOUR_KEY")
            return False
    
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Ошибка: google-generativeai не установлен")
        print("   Установите: pip install google-generativeai")
        return False
    
    print("🔍 Проверка доступности Gemini API...")
    print(f"   API ключ: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}\n")
    
    try:
        # Настраиваем API
        genai.configure(api_key=api_key)
        
        # Получаем список моделей
        print("📋 Получение списка доступных моделей...")
        all_models = genai.list_models()
        
        # Фильтруем модели с поддержкой generateContent
        available_models = [
            m for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not available_models:
            print("❌ Нет доступных моделей с поддержкой generateContent")
            return False
        
        print(f"\n✅ API доступен! Найдено {len(available_models)} моделей:\n")
        
        # Группируем модели
        flash_models = []
        pro_models = []
        other_models = []
        
        for m in available_models:
            name = m.name.replace("models/", "")
            if "flash" in name.lower():
                flash_models.append(name)
            elif "pro" in name.lower():
                pro_models.append(name)
            else:
                other_models.append(name)
        
        if flash_models:
            print("⚡ Flash модели (быстрые):")
            for m in flash_models:
                print(f"   - {m}")
            print()
        
        if pro_models:
            print("🎯 Pro модели (точные):")
            for m in pro_models:
                print(f"   - {m}")
            print()
        
        if other_models:
            print("📦 Другие модели:")
            for m in other_models:
                print(f"   - {m}")
            print()
        
        # Тестовый запрос
        print("🧪 Тестовый запрос...")
        test_model = flash_models[0] if flash_models else (pro_models[0] if pro_models else available_models[0].name.replace("models/", ""))
        print(f"   Используем модель: {test_model}")
        
        try:
            model_instance = genai.GenerativeModel(test_model)
            response = model_instance.generate_content(
                "Say hello in one word.",
                generation_config={"temperature": 0.1, "max_output_tokens": 50}
            )
            
            # Проверяем finish_reason
            if not response.candidates:
                print("   ⚠️  Нет кандидатов в ответе")
                print("   ✅ Но API доступен (список моделей получен успешно)")
            else:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                finish_reason_name = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                
                # Пытаемся получить текст
                try:
                    answer_text = response.text.strip()
                    if answer_text:
                        print(f"   Ответ модели: {answer_text}")
                        if finish_reason_name == "STOP" or finish_reason == 1:
                            print("\n✅ Gemini API работает в вашем регионе!")
                        elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                            print("\n✅ Gemini API работает (ответ обрезан из-за лимита токенов)")
                        else:
                            print(f"\n✅ Gemini API работает (finish_reason: {finish_reason_name})")
                    else:
                        # Нет текста, но есть finish_reason
                        if finish_reason_name == "SAFETY" or finish_reason == 3:
                            print("   ⚠️  Ответ заблокирован safety filter")
                            print("   ✅ API работает, просто этот запрос был заблокирован")
                        elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                            print("   ⚠️  Ответ обрезан из-за лимита токенов")
                            print("   ✅ API работает")
                        else:
                            print(f"   ⚠️  Finish reason: {finish_reason_name} ({finish_reason})")
                            print("   ✅ API работает")
                except (ValueError, AttributeError) as e:
                    # Не удалось получить текст
                    if finish_reason_name == "SAFETY" or finish_reason == 3:
                        print("   ⚠️  Ответ заблокирован safety filter")
                        print("   ✅ API работает, просто этот запрос был заблокирован")
                    elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                        print("   ⚠️  Ответ обрезан из-за лимита токенов (нет текста)")
                        print("   ✅ API работает")
                    else:
                        print(f"   ⚠️  Не удалось получить текст (finish_reason: {finish_reason_name})")
                        print("   ✅ Но API доступен (список моделей получен успешно)")
        except Exception as e:
            print(f"   ⚠️  Ошибка тестового запроса: {e}")
            print("   ✅ Но API доступен (список моделей получен успешно)")
        
        print(f"\n💡 Рекомендуемая модель для использования: {flash_models[0] if flash_models else pro_models[0] if pro_models else 'любая доступная'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при обращении к Gemini API:")
        print(f"   {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Тест доступности Gemini API")
    parser.add_argument("--api-key", help="Gemini API ключ (или используйте переменную GEMINI_API_KEY)")
    args = parser.parse_args()
    
    success = test_gemini_api(api_key=args.api_key)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

