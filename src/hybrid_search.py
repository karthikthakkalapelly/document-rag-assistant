from rank_bm25 import BM25Okapi


class HybridSearch:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def keyword_search(
        self,
        query,
        k=5
    ):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(
                self.documents,
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:k]