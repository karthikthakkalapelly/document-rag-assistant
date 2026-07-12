import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    print("========== DEBUG ==========")
    print("API key exists:", api_key is not None)
    print("API key length:", len(api_key) if api_key else 0)
    print("===========================")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    return llm