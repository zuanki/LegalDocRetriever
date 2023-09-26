from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

class LexicalRetrieval(ABC):
    def __init__(
        self,
        data: List[Dict[str, str]]
    ):
        self.data = data

    def data2documents(self):
        """
        Parse the data to documents
        :return: documents
        """
        documents = []
        for law in self.data:
            for article in law['articles']:
                documents.append(article['segmented_text'])
        return documents
    
    def info_search(
        self,
        document: str
    ) -> Dict[str, str]:
        """
        Search for the law that contains the document
        :param document: document
        :return: law that contains the document
        """

        res = {}
        for law in self.data:
            for article in law['articles']:
                if article['segmented_text'] == document:
                    res['law_id'] = law['id']
                    res['article_id'] = article['id']
                    # res["text"] = article["text"]
                    # res["segmented_text"] = article["segmented_text"]

        return res
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        """
        Retrieve top_k documents
        :param query: query text
        :param top_k: number of documents to retrieve
        :return: top_k documents
        """
        pass

    @abstractmethod
    def get_scores(
        self,
        query_tokens: List[str]
    ) -> np.ndarray:
        pass

    def score(
        self,
        query: str,
        document: str
    ) -> float:
        # Check if document is in corpus
        assert document in self.documents, "Document not in corpus"

        # Split query into tokens
        query_tokens = query.split()

        # Calculate scores
        scores = self.get_scores(query_tokens)

        # Min max scaling to [0, 1]
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        for i, doc in enumerate(self.documents):
            if doc == document:
                return scores[i]
            
        return 0.0