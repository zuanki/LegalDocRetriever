# Convert json to csv
from typing import List, Dict, Any
import pandas as pd

def json2csv(data: List[Dict[str, Any]], output_path: str):
    """
    json format:
    [
        {
            "question_id": "q-1",
            "text": "Chiếm đoạt di vật của tử sĩ có thể bị phạt tù lên đến bao nhiêu năm?",
            "segmented_text": "chiếm_đoạt di_vật của tử_sĩ có_thể bị phạt tù lên đến bao_nhiêu năm",
            "answer": "07 năm",
            "relevant_articles": [
                {
                    "law_id": "Văn bản hợp nhất",
                    "article_id": "418",
                    "text": "Tội chiếm đoạt hoặc hủy hoại di vật của tử sỹ\n\n1. Người nào chiếm đoạt hoặc hủy hoại di vật của tử sỹ, thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm.\n\n2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 02 năm đến 07 năm:\n\na) Là chỉ huy hoặc sĩ quan;\n\nb) Chiếm đoạt hoặc hủy hoại di vật của 02 tử sỹ trở lên.",
                    "segmented_text": "tội chiếm_đoạt hoặc huỷ_hoại di_vật của tử_sỹ người nào chiếm_đoạt hoặc huỷ_hoại di_vật của tử_sỹ thì bị phạt cải_tạo không giam_giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm phạm_tội thuộc một trong các trường_hợp sau đây thì bị phạt tù từ 02 năm đến 07 năm là chỉ_huy hoặc sĩ_quan chiếm_đoạt hoặc huỷ_hoại di_vật của 02 tử_sỹ trở lên"
                }
            ]
        },
        ...
    ]
    """

    res = []

    for item in data[:10]: # TODO: remove this
        for relevant_article in item["relevant_articles"]:
            res.append({
                "query": item["text"],
                "document": relevant_article["text"],
                "label": 1
            })

    df = pd.DataFrame(res)

    df.to_csv(output_path, index=False)