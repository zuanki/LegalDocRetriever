import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexical_base import LexicalRetrieval

from typing import List, Dict


class TFIDFRetrieval(LexicalRetrieval):
    def __init__(
        self,
        data: List[Dict[str, str]]
    ):
        super().__init__(data)

        self.documents = self.data2documents()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        query = " ".join(query_tokens)
        query_vector = self.tfidf_vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vector.T).toarray().flatten()
        return scores

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        scores = self.get_scores(query.split())
        sorted_scores = np.argsort(scores)[::-1][:top_k]

        # Min-max scaling to [0, 1]
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        res = []
        for i in sorted_scores:
            tmp = self.info_search(self.documents[i])
            tmp["tfidf_score"] = scores[i]
            res.append(tmp)

        return res
