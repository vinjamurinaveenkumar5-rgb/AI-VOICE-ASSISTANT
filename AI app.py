import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from environment variables.")

client = Groq(api_key=api_key)

def generate_ai_response(transcribed_text):
    # Standard active chat completion models on Groq
    preferred_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "mixtral-8x7b-32768"
    ]
    
    # 1. First, attempt preferred standard chat models directly
    for model_name in preferred_models:
        try:
            print(f"Attempting model: {model_name}...")
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": transcribed_text}],
                max_tokens=500,
            )
            print(f"Success! Response generated using: {model_name}")
            return chat_completion.choices[0].message.content
        except Exception as err:
            print(f"Model {model_name} skipped: {err}")
            continue

    # 2. Fallback: Query account for any active standard text model
    try:
        models_data = client.models.list().data
        for model in models_data:
            m_id = model.id.lower()
            # Exclude speech, audio, safety, guard, and third-party models
            if any(x in m_id for x in ["whisper", "orpheus", "guard", "canopylabs", "vision"]):
                continue
            
            try:
                print(f"Attempting dynamic account model: {model.id}...")
                chat_completion = client.chat.completions.create(
                    model=model.id,
                    messages=[{"role": "user", "content": transcribed_text}],
                    max_tokens=500,
                )
                print(f"Success! Response generated using: {model.id}")
                return chat_completion.choices[0].message.content
            except Exception as err:
                print(f"Dynamic model {model.id} failed: {err}")
                continue
    except Exception as list_err:
        print(f"Failed to fetch model list: {list_err}")

    raise RuntimeError("All model attempts failed. Please verify your GROQ_API_KEY on console.groq.com.")

# Test code block
if __name__ == "__main__":
    test_input = "Hello, can you hear me?"
    print(f"Testing input: '{test_input}'")
    try:
        response = generate_ai_response(test_input)
        print(f"\nAI Output:\n{response}")
    except Exception as e:
        print(f"\nError encountered: {e}")