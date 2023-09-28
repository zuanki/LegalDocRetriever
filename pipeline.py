# Input: Query
# Output: List of relevant documents
# Description: This file contains the pipeline for the search engine.

# Importing libraries
from typing import List, Dict, Any
import json
from src.models.lexical import BM25Retrieval
from src.preprocessing.utils import clean_query
import py_vncorenlp


def pipeline(query: str) -> List[Dict[str, Any]]:
    """
    This function is the pipeline for the search engine.
    :param query: Query string
    :return: List of relevant documents
    """
    # Search Space
    with open("data/BM25/2023/law.json", "r", encoding="utf-8") as f:
        search_space = json.load(f)

    # BM25 Retrieval
    bm25 = BM25Retrieval(search_space)
    rdrsegmenter = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], save_dir='/home/zuanki/Project/LegalDocRetriever/VnCoreNLP')

    # Process query
    cleaned_query = clean_query(query)
    segmented_query = rdrsegmenter.word_segment(cleaned_query)[0]

    # BM25 Search
    relevant_documents = bm25.retrieve(segmented_query, top_k=3)

    # BERT
    # TODO

    return relevant_documents


if __name__ == "__main__":
    query = "Tội chiếm đoạt hoặc hủy hoại di vật của tử sỹ"
    relevant_documents = pipeline(query)
    print(relevant_documents)
