import os
from collections import defaultdict


class RAGPipeline:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.hybrid_search = None
        self.pdf_names = []
        self.ocr_documents = []
        self.total_pages = 0
        self.total_chunks = 0
        self.database_path = None

    def load_llm_if_needed(self):
        if self.llm is None:
            from src.llm import load_llm
            self.llm = load_llm()
        return self.llm

    def build_database(self, pdf_paths, embedding_model=None, **kwargs):
        from src.chunker import create_chunks
        from src.pdf_loader import load_pdf
        from src.hybrid_search import HybridSearch
        from src.vector_store import create_vector_store

        all_documents = []
        self.pdf_names = []
        self.ocr_documents = []
        self.total_pages = 0
        self.total_chunks = 0

        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        for pdf_path in pdf_paths:
            documents, ocr_used = load_pdf(pdf_path)
            if ocr_used:
                self.ocr_documents.append(os.path.basename(pdf_path))
            self.pdf_names.append(os.path.basename(pdf_path))
            self.total_pages += len(documents)
            all_documents.extend(documents)

        chunks = create_chunks(all_documents)
        self.total_chunks = len(chunks)
        self.hybrid_search = HybridSearch(chunks)

        if embedding_model is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                google_api_key=api_key,
            )

        self.database_path = os.path.join(
            "database",
            str(hash(tuple(sorted(self.pdf_names))))
        )
        # create and persist the vector store, keep reference for queries
        self.vector_store = create_vector_store(
            chunks, self.database_path, embedding_model=embedding_model
        )
        return self.database_path

    def ask(self, question):
        if self.vector_store is None:
            return ("Vector store is not initialized.", [], 0)

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10},
        )

        documents = list(retriever.invoke(question))
        combined_documents = list(documents)

        keyword_results = self.hybrid_search.keyword_search(question, k=3)
        seen = {
            (doc.metadata["source"], doc.metadata["page"]) for doc in combined_documents
        }
        for document, _score in keyword_results:
            key = (document.metadata["source"], document.metadata["page"])
            if key not in seen:
                combined_documents.append(document)
                seen.add(key)

        documents = combined_documents

        if not documents:
            return (
                "I couldn't find any relevant information in the uploaded documents.",
                [],
                0,
            )

        if len(documents) < 5:
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

            seen = set()
            documents = []
            for doc in expanded_documents:
                key = (doc.metadata["source"], doc.metadata["page"], doc.page_content)
                if key not in seen:
                    seen.add(key)
                    documents.append(doc)

        documents = documents[:8]

        retrieved_pdf_names = list(
            {os.path.basename(doc.metadata["source"]) for doc in documents}
        )

        document_groups = defaultdict(list)
        for document in documents:
            pdf_name = os.path.basename(document.metadata["source"])
            document_groups[pdf_name].append(document)

        context_parts = []
        for pdf_name, docs in document_groups.items():
            context_parts.append(f"========== {pdf_name} ==========")
            for document in docs:
                page = document.metadata["page"]
                context_parts.append(
                    f"[Document: {pdf_name} | Page: {page}] {document.page_content}"
                )

        context = "\n".join(context_parts)

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
2. If only one document is relevant, answer only from that document.
3. If multiple documents are relevant, compare them.
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

        llm = self.load_llm_if_needed()
        if llm is None:
            return (
                "The AI model is not available because the cloud API key is not configured. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in Render environment variables.",
                documents,
                0,
            )

        response = llm.invoke(prompt)

        if len(documents) >= 15:
            confidence = 95
        elif len(documents) >= 10:
            confidence = 90
        elif len(documents) >= 5:
            confidence = 85
        else:
            confidence = 75

        return (response.content, documents, confidence)
