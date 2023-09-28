import streamlit as st

from typing import List, Dict, Any
import json
from src.models.lexical import BM25Retrieval
from src.preprocessing.utils import clean_query
from underthesea import word_tokenize

# Search Space
with open("/home/zuanki/Project/LegalDocRetriever/data/BM25/2023/law.json", "r", encoding="utf-8") as f:
    search_space = json.load(f)

# BM25 Retrieval
bm25 = BM25Retrieval(search_space)


def pipeline(query: str) -> List[Dict[str, Any]]:
    """
    This function is the pipeline for the search engine.
    :param query: Query string
    :return: List of relevant documents
    """

    # Process query
    cleaned_query = clean_query(query)
    segmented_query = word_tokenize(cleaned_query, format="text")

    # BM25 Search
    relevant_documents = bm25.retrieve(segmented_query, top_k=3)

    # BERT
    # TODO

    return relevant_documents


def main():
    # Add style to your Streamlit app
    st.markdown("""
    <style>
        .reportview-container {
            background-color: #f4f4f4;
        }
    </style>
    """, unsafe_allow_html=True)

    # Set the title of your Streamlit app
    st.markdown(
        f"<span style='color: #42d68c; font-size:40px; font-weight:bold;'>Legal Document Retrieval</span>",
        unsafe_allow_html=True
    )

    # Using a form for the input field and button
    with st.form(key='search_form'):
        # Retrieve user input
        user_input = st.text_input("Enter your query here:")
        # Create a submit button inside the form with the label 'Search'
        submitted = st.form_submit_button('Search')

    # Check if the form was submitted
    if submitted and user_input:
        # Call your document retrieval pipeline (ensure you have this function defined)
        relevant_documents = pipeline(user_input)

        # Display retrieved documents
        st.subheader("Retrieved Documents:")

        # Iterate over the list of retrieved documents
        for doc in relevant_documents:
            # Format of a document: {"law_id": str, "article_id": str, "text": str}
            # Create a horizontal rule with a custom color
            st.markdown("<hr style='border: 2px solid #42d68c'>",
                        unsafe_allow_html=True)

            # Display law_id and article_id with custom styling
            st.markdown(
                f"<span style='color: #42d68c; font-size:20px; font-weight:bold;'>{doc['law_id']} - Điều {doc['article_id']}</span>",
                unsafe_allow_html=True
            )

            # Display the text of the document
            st.write(doc["text"])


if __name__ == "__main__":
    main()
