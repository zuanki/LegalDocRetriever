from rank_bm25 import BM25Plus
import numpy as np
from typing import List, Dict

from ..lexical_base import LexicalRetrieval

class BM25Retrieval(LexicalRetrieval):
    def __init__(
        self,
        data: List[Dict[str, str]]
    ):
        super().__init__(data)


        self.documents = self.data2documents()
        self.tokenize_documents = [doc.split() for doc in self.documents]
        
        self.bm25 = BM25Plus(
            corpus=self.tokenize_documents,
            k1=1.2,
            b=0.75,
            delta=1
        )


    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        return self.bm25.get_scores(query_tokens)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        scores = self.get_scores(query.split())
        sorted_scores = np.argsort(scores)[::-1][:top_k]

        # Min-max scaling to [0, 1]
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        res = []
        for i in sorted_scores:
            tmp = self.info_search(self.documents[i])
            tmp["bm25_score"] = scores[i]
            res.append(tmp)

        return res