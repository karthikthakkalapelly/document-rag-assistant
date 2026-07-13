import os
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    print("========== DEBUG ==========")
    print("API key exists:", api_key is not None)
    print("API key length:", len(api_key) if api_key else 0)
    print("===========================")

    if not api_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        print("ERROR: Could not import langchain_google_genai:", exc)
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    return llm