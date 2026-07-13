def load_vector_store(database_path, embedding_model=None):
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=database_path,
        embedding_function=embedding_model,
    )


def search_documents(vector_store, query, k=3):
    results = vector_store.similarity_search_with_score(query=query, k=k)
    return results
