#!/usr/bin/env python3
"""Test Gemini API availability and list models."""

import os
import sys

def test_gemini_api(api_key: str = None):
    """Test Gemini API availability and show available models."""
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: Gemini API key not specified")
            print("   Use: export GEMINI_API_KEY='your-key'")
            print("   Or pass via parameter: python test_gemini.py --api-key YOUR_KEY")
            return False
    
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: google-generativeai not installed")
        print("   Install: pip install google-generativeai")
        return False
    
    print("Checking Gemini API availability...")
    print(f"   API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}\n")
    
    try:
        genai.configure(api_key=api_key)
        
        print("Getting list of available models...")
        all_models = genai.list_models()
        
        available_models = [
            m for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not available_models:
            print("No available models with generateContent support")
            return False
        
        print(f"\nAPI available! Found {len(available_models)} models:\n")
        
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
            print("Flash models (fast):")
            for m in flash_models:
                print(f"   - {m}")
            print()
        
        if pro_models:
            print("Pro models (accurate):")
            for m in pro_models:
                print(f"   - {m}")
            print()
        
        if other_models:
            print("Other models:")
            for m in other_models:
                print(f"   - {m}")
            print()
        
        print("Test query...")
        test_model = flash_models[0] if flash_models else (pro_models[0] if pro_models else available_models[0].name.replace("models/", ""))
        print(f"   Using model: {test_model}")
        
        try:
            model_instance = genai.GenerativeModel(test_model)
            response = model_instance.generate_content(
                "Say hello in one word.",
                generation_config={"temperature": 0.1, "max_output_tokens": 50}
            )
            
            if not response.candidates:
                print("   No candidates in response")
                print("   But API is available (model list retrieved successfully)")
            else:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                finish_reason_name = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                
                try:
                    answer_text = response.text.strip()
                    if answer_text:
                        print(f"   Model response: {answer_text}")
                        if finish_reason_name == "STOP" or finish_reason == 1:
                            print("\nGemini API is working in your region!")
                        elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                            print("\nGemini API is working (response truncated due to token limit)")
                        else:
                            print(f"\nGemini API is working (finish_reason: {finish_reason_name})")
                    else:
                        if finish_reason_name == "SAFETY" or finish_reason == 3:
                            print("   Response blocked by safety filter")
                            print("   API is working, just this query was blocked")
                        elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                            print("   Response truncated due to token limit")
                            print("   API is working")
                        else:
                            print(f"   Finish reason: {finish_reason_name} ({finish_reason})")
                            print("   API is working")
                except (ValueError, AttributeError) as e:
                    if finish_reason_name == "SAFETY" or finish_reason == 3:
                        print("   Response blocked by safety filter")
                        print("   API is working, just this query was blocked")
                    elif finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                        print("   Response truncated due to token limit (no text)")
                        print("   API is working")
                    else:
                        print(f"   Failed to get text (finish_reason: {finish_reason_name})")
                        print("   But API is available (model list retrieved successfully)")
        except Exception as e:
            print(f"   Test query error: {e}")
            print("   But API is available (model list retrieved successfully)")
        
        print(f"\nRecommended model: {flash_models[0] if flash_models else pro_models[0] if pro_models else 'any available'}")
        
        return True
        
    except Exception as e:
        print(f"\nError calling Gemini API:")
        print(f"   {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gemini API availability")
    parser.add_argument("--api-key", help="Gemini API key (or use GEMINI_API_KEY env var)")
    args = parser.parse_args()
    
    success = test_gemini_api(api_key=args.api_key)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

