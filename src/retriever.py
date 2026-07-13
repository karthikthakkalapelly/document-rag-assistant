def load_vector_store(database_path):   # Loads the existing chroma database
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=database_path,
        embedding_function=embedding_model,
    )
    return vector_store


def search_documents(vector_store, query, k=3):
    # Searches for the most relevant chunks
    results = vector_store.similarity_search_with_score(query=query, k=k)
    return results
