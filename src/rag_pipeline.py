from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.vector_store import create_vector_store
from src.retriever import load_vector_store
from src.llm import load_llm
import os
from collections import defaultdict
from src.hybrid_search import HybridSearch

class RAGPipeline:
    def __init__(self):
        self.vector_store=None
        self.llm=load_llm()
        self.hybrid_search = None

        #store document information
        self.pdf_names=[]
        self.ocr_documents = []
        self.total_pages=0
        self.total_chunks=0
        

    def build_database(self,pdf_paths):
        #Build or load the vector database for using multiple PDF's
        all_documents=[]
        #Reset information everytime  a new set of PDF S is uploaded
        self.pdf_names = []
        self.ocr_documents = []
        self.total_pages = 0
        self.total_chunks = 0

        #Read every uploaded PDF

        from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.vector_store import create_vector_store
from src.retriever import load_vector_store
from src.llm import load_llm
import os
from collections import defaultdict
from src.hybrid_search import HybridSearch

class RAGPipeline:
    def __init__(self):
        self.vector_store=None
        self.llm=load_llm()
        self.hybrid_search = None

        #store document information
        self.pdf_names=[]
        self.ocr_documents = []
        self.total_pages=0
        self.total_chunks=0
        

    def build_database(self,pdf_paths):
        #Build or load the vector database for using multiple PDF's
        all_documents=[]
        #Reset information everytime  a new set of PDF S is uploaded
        self.pdf_names = []
        self.ocr_documents = []
        self.total_pages = 0
        self.total_chunks = 0

        #Read every uploaded PDF

        for pdf_path in pdf_paths:
            print(f"Loading: {pdf_path}")

            documents, ocr_used = load_pdf(pdf_path)

            if ocr_used:
                print(f"OCR used for {os.path.basename(pdf_path)}")
                self.ocr_documents.append(os.path.basename(pdf_path))

    # ALWAYS execute
            self.pdf_names.append(os.path.basename(pdf_path))
            self.total_pages += len(documents)
            all_documents.extend(documents)

        print(f"Total Pages:{self.total_pages}")

        #Create chunks
        chunks=create_chunks(all_documents)
        self.total_chunks=len(chunks)
        # Build BM25 keyword index
        self.hybrid_search = HybridSearch(chunks)
        print(f"Total chunks:{self.total_chunks}")

        #Build database
        database_path=os.path.join("database",str(hash(tuple(sorted(self.pdf_names)))))
        create_vector_store(chunks,database_path)
        self.vector_store = load_vector_store(
            database_path
        )

        print("Vector database created successfully.")


    def ask(self,question):
        
        
        #vector search
        
        retriever = self.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":20,
        "fetch_k":60
    }
)

        documents = retriever.invoke(question)
        combined_documents = list(documents)
        #keyword_search
        keyword_results = self.hybrid_search.keyword_search(
        question,k=15
        )
        #Merge results
        
            
        # Add Keyword Results
        seen = {
        (
        doc.metadata["source"],
        doc.metadata["page"]
        )
        for doc in combined_documents
        }
        for document, score in keyword_results:
            key = (
        document.metadata["source"],
        document.metadata["page"]
              )

            if key not in seen:
                combined_documents.append(document)
               
                seen.add(key)

        documents = combined_documents
        print("\nRetrieved Documents:")

        for doc in documents:
            print(
        os.path.basename(doc.metadata["source"]),
        "Page",
        doc.metadata["page"]
    )
        # Documents involved in retrieval
        retrieved_pdf_names = list(
    {
        os.path.basename(doc.metadata["source"])
        for doc in documents
    }
)
       
        if not documents:
            return (
        "I couldn't find any relevant information in the uploaded documents.",
        [],
        0
         )  
        # ---------------------------------------
# Expand retrieved chunks with neighboring chunks
# ---------------------------------------

        expanded_documents = []
        for doc in documents:
            expanded_documents.append(doc)
            current_page = doc.metadata["page"]
            source = doc.metadata["source"]
            for candidate in self.hybrid_search.documents:
                if (
            candidate.metadata["source"] == source
            and abs(candidate.metadata["page"] - current_page) == 1
        ):
                    expanded_documents.append(candidate)

# Remove duplicates
        seen = set()
        documents = []

        for doc in expanded_documents:
            key = (
        doc.metadata["source"],
        doc.metadata["page"],
        doc.page_content
    )

            if key not in seen:
                seen.add(key)
                documents.append(doc)     

        print("MMR Retrieval Completed")
        

       
        # ---------------------------------------
        # Group retrieved chunks by document
        # ---------------------------------------

        document_groups = defaultdict(list)

        for document in documents:
            pdf_name = os.path.basename(
            document.metadata["source"]
    )

            document_groups[pdf_name].append(document)
        print("\nRetrieved Documents")


        for pdf_name, docs in document_groups.items():
            print(f"\n{pdf_name}")
            for doc in docs:
                print(f"  Page {doc.metadata['page']}")

# ---------------------------------------
# Build context document-wise
# ---------------------------------------

        context = ""

        for pdf_name, docs in document_groups.items():
            context += f"\n========== {pdf_name} ==========\n\n"

            for document in docs:
                page = document.metadata["page"]

                context += (
                f"[Document: {pdf_name} | Page: {page}]\n"
                f"{document.page_content}\n\n"
                  )
        

        prompt = f"""
You are an expert Multi-Document AI Assistant.

The uploaded documents are:

{retrieved_pdf_names}

These are the ONLY documents available.

Determine which documents are relevant before answering.
If two or more documents are relevant, compare them.
If only one document is relevant, answer only from that document.
Never ignore a retrieved document.



Below is the retrieved context from those documents.

-----------------------
{context}
-----------------------

Question:
{question}

Rules:

1. First determine which document(s) contain the answer.

2. If only one document is relevant,
answer only from that document.

3. If multiple documents are relevant,
compare them.

4. For comparisons, always produce:

# Overview

# Similarities

# Differences

# Conclusion

5. Never mix information from unrelated PDFs.

6. Never invent facts.

7. Mention the filename whenever information comes from a document.

8. If the answer cannot be found, reply:

"I couldn't find that information in the uploaded documents."

Answer:
"""
       
        response=self.llm.invoke(prompt)
        
        print("\n==============================")
        print("Retrieved Documents:", len(documents))
        print("==============================")

        if len(documents) >= 15:
            confidence = 95
        elif len(documents) >= 10:
            confidence = 90
        elif len(documents) >= 5:
            confidence = 85
        else:
            confidence = 75
        

        return (response.content,documents,confidence)

        print(f"Total Pages:{self.total_pages}")

        #Create chunks
        chunks=create_chunks(all_documents)
        self.total_chunks=len(chunks)
        # Build BM25 keyword index
        self.hybrid_search = HybridSearch(chunks)
        print(f"Total chunks:{self.total_chunks}")

        #Build database
        database_path=os.path.join("database",str(hash(tuple(sorted(self.pdf_names)))))
        create_vector_store(chunks,database_path)
        self.vector_store = load_vector_store(
            database_path
        )

        print("Vector database created successfully.")


    def ask(self,question):
        
        
        #vector search
        
        retriever = self.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":20,
        "fetch_k":60
    }
)

        documents = retriever.invoke(question)
        combined_documents = list(documents)
        #keyword_search
        keyword_results = self.hybrid_search.keyword_search(
        question,k=15
        )
        #Merge results
        
            
        # Add Keyword Results
        seen = {
        (
        doc.metadata["source"],
        doc.metadata["page"]
        )
        for doc in combined_documents
        }
        for document, score in keyword_results:
            key = (
        document.metadata["source"],
        document.metadata["page"]
              )

            if key not in seen:
                combined_documents.append(document)
               
                seen.add(key)

        documents = combined_documents
        print("\nRetrieved Documents:")

        for doc in documents:
            print(
        os.path.basename(doc.metadata["source"]),
        "Page",
        doc.metadata["page"]
    )
        # Documents involved in retrieval
        retrieved_pdf_names = list(
    {
        os.path.basename(doc.metadata["source"])
        for doc in documents
    }
)
       
        if not documents:
            return (
        "I couldn't find any relevant information in the uploaded documents.",
        [],
        0
         )  
        # ---------------------------------------
# Expand retrieved chunks with neighboring chunks
# ---------------------------------------

        expanded_documents = []
        for doc in documents:
            expanded_documents.append(doc)
            current_page = doc.metadata["page"]
            source = doc.metadata["source"]
            for candidate in self.hybrid_search.documents:
                if (
            candidate.metadata["source"] == source
            and abs(candidate.metadata["page"] - current_page) == 1
        ):
                    expanded_documents.append(candidate)

# Remove duplicates
        seen = set()
        documents = []

        for doc in expanded_documents:
            key = (
        doc.metadata["source"],
        doc.metadata["page"],
        doc.page_content
    )

            if key not in seen:
                seen.add(key)
                documents.append(doc)     

        print("MMR Retrieval Completed")
        

       
        # ---------------------------------------
        # Group retrieved chunks by document
        # ---------------------------------------

        document_groups = defaultdict(list)

        for document in documents:
            pdf_name = os.path.basename(
            document.metadata["source"]
    )

            document_groups[pdf_name].append(document)
        print("\nRetrieved Documents")


        for pdf_name, docs in document_groups.items():
            print(f"\n{pdf_name}")
            for doc in docs:
                print(f"  Page {doc.metadata['page']}")

# ---------------------------------------
# Build context document-wise
# ---------------------------------------

        context = ""

        for pdf_name, docs in document_groups.items():
            context += f"\n========== {pdf_name} ==========\n\n"

            for document in docs:
                page = document.metadata["page"]

                context += (
                f"[Document: {pdf_name} | Page: {page}]\n"
                f"{document.page_content}\n\n"
                  )
        

        prompt = f"""
You are an expert Multi-Document AI Assistant.

The uploaded documents are:

{retrieved_pdf_names}

These are the ONLY documents available.

Determine which documents are relevant before answering.
If two or more documents are relevant, compare them.
If only one document is relevant, answer only from that document.
Never ignore a retrieved document.



Below is the retrieved context from those documents.

-----------------------
{context}
-----------------------

Question:
{question}

Rules:

1. First determine which document(s) contain the answer.

2. If only one document is relevant,
answer only from that document.

3. If multiple documents are relevant,
compare them.

4. For comparisons, always produce:

# Overview

# Similarities

# Differences

# Conclusion

5. Never mix information from unrelated PDFs.

6. Never invent facts.

7. Mention the filename whenever information comes from a document.

8. If the answer cannot be found, reply:

"I couldn't find that information in the uploaded documents."

Answer:
"""
       
        response=self.llm.invoke(prompt)
        
        print("\n==============================")
        print("Retrieved Documents:", len(documents))
        print("==============================")

        if len(documents) >= 15:
            confidence = 95
        elif len(documents) >= 10:
            confidence = 90
        elif len(documents) >= 5:
            confidence = 85
        else:
            confidence = 75
        

        return (response.content,documents,confidence)