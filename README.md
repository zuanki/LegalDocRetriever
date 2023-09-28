# Legal Document Retrieval

The primary objective of this project is to retrieve Vietnamese legal documents based on user-provided queries. The retrieval process is divided into three key stages:

1. **First Stage Retrieval:** Utilizing [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
   - In this initial stage, the BM25 algorithm is employed to swiftly narrow down the search scope and present a collection of potentially relevant documents.

2. **Second Stage Retrieval:** Fine-tuning [BERT](https://arxiv.org/abs/1810.04805) with [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)
    - In this phase, the BERT model is enhanced through the application of the LoRA adapter layer. This innovative approach to fine-tuning BERT is especially valuable for low-resource languages like Vietnamese. It facilitates the fine-tuning process on a small dataset without the necessity for a large corpus.

3. **Ensemble of BM25 and BERT:** Enhanced Retrieval
   - The final results are achieved through an ensemble approach, combining the outcomes from both stages. This synergistic approach capitalizes on the unique strengths of each model, resulting in a robust retrieval system proficient in comprehending both lexical and semantic aspects of the query.

## Vietnamese Legal Document Dataset
**Sample Query:**
```
Trường hợp công dân không trong độ tuổi nhập ngũ, nếu đi du học, xuất khẩu lao động không cần phải khai báo tạm vắng, đúng hay sai?
```
(English Translation: "In the case of citizens who are not of military age, if they study abroad or export labor, they do not need to declare temporary absence. Is it true or false?")

**Output:**
```json
"relevant_articles": [
    {
        "law_id": "Luật Cư trú",
        "article_id": "31"
    }
]
```
## Experiment Results
![image](https://github.com/zuanki/LegalDocRetriever/blob/main/assets/exp.png)

## Streamlit App
![image](https://github.com/zuanki/LegalDocRetriever/blob/main/assets/demo.png)