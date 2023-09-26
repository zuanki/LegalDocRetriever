import math
import numpy as np
from nltk.probability import FreqDist

from typing import List, Dict

from ..lexical_base import LexicalRetrieval

class QLDRetrieval(LexicalRetrieval):
    def __init__(
        self,
        data: List[Dict[str, str]]
    ):
        super().__init__(data)

        self.documents = self.data2documents()

        # Hyper-parameters
        self.n = 100
        self.alpha_d = 0.1
        self.epsilon = 0.00001
        self.tokenized_docs, self.fdist_docs = self.preprocess_documents()
        self.collection_fdist = self.compute_collection_frequencies()

    def preprocess_documents(self) -> List[List[str]]:
        tokenized_docs = [doc.split() for doc in self.documents]
        fdist_docs = [FreqDist(doc) for doc in tokenized_docs]
        return tokenized_docs, fdist_docs
    
    def compute_collection_frequencies(self) -> FreqDist:
        all_tokens = [token for doc in self.tokenized_docs for token in doc]
        return FreqDist(all_tokens)

    def compute_term_scores(self, query_tf: FreqDist, doc_tf: FreqDist) -> List[float]:
        term_scores = []
        
        for term in query_tf:
            if term in doc_tf:
                p_qi_d = (doc_tf[term] / len(doc_tf)) / self.alpha_d
                p_qi_c = (self.collection_fdist[term] / len(self.collection_fdist))
                term_scores.append(math.log(p_qi_d / p_qi_c + self.epsilon))

        return term_scores
    
    def compute_qld_scores(self, query_tf: FreqDist, doc_tfs: List[FreqDist]) -> List[float]:
        scores = []
        for doc_tf in doc_tfs:
            term_scores = self.compute_term_scores(query_tf, doc_tf)
            doc_score = np.sum(term_scores)
            doc_score += self.n * math.log(self.alpha_d)
            doc_score += np.sum([math.log(self.collection_fdist[term] / len(self.collection_fdist) + self.epsilon) for term in query_tf])
            scores.append(doc_score)
        return scores
    
    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        query_tf = FreqDist(query_tokens)
        doc_tfs = [FreqDist(doc) for doc in self.tokenized_docs]

        scores = self.compute_qld_scores(query_tf, doc_tfs)
        return scores
    
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
            tmp["qld_score"] = scores[i]
            res.append(tmp)

        return res