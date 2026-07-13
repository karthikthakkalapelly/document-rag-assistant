import os

def create_vector_store(chunks, database_path, embedding_model=None):
    from langchain_chroma import Chroma

    if embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    os.makedirs(database_path, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=database_path,
    )
    return vector_store
