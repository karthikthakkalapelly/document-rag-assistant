import os

def create_vector_store(chunks, database_path, embedding_model=None):
    from langchain_chroma import Chroma

    if embedding_model is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )

    os.makedirs(database_path, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=database_path,
    )
    return vector_store
