from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents):   #splits text into smaller chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    return splitter.split_documents(documents)
