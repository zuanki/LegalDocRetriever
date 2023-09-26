# Rename
# import json
# from tqdm import tqdm

# with open("./data/BM25/2023/train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# new_data = []

# for question in tqdm(data):
#     tmp = {}
#     tmp["question_id"] = question["question_id"]
#     tmp["text"] = question["text"]
#     tmp["segmented_text"] = question["bm25_text"]
#     tmp["answer"] = question["answer"]
#     tmp["relevant_articles"] = []

#     for article in question["relevant_articles"]:
#         tmp["relevant_articles"].append({
#             "law_id": article["law_id"],
#             "article_id": article["article_id"],
#             "text": article["text"],
#             "segmented_text": article["bm25_text"]
#         })

#     new_data.append(tmp)
    

# # Save to file
# with open("./data/BM25/2023/new_train.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)

# from transformers import AutoTokenizer, AutoModel
# from src.models.bert_cls import BertCLS
# from src.datasets.law_dataset import LawDataset

# import pandas as pd

# model_name = "checkpoints/m_bert"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # bert_model = AutoModel.from_pretrained(model_name)

# # model = BertCLS(bert_model)

# dataset = LawDataset(
#     df=pd.read_csv("data/BM25/2022/train.csv"),
#     tokenizer=tokenizer,
#     max_len=512,
#     train=True
# )

# print(dataset[0])

# print(model)

# sentence = "Tôi là sinh_viên trường đại_học bách_khoa hà_nội"

# encoding = tokenizer(
#     sentence,
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
#     max_length=128
# )

# logits = model(
#     input_ids=encoding["input_ids"],
#     attention_mask=encoding["attention_mask"],
#     token_type_ids=encoding["token_type_ids"]
# )

# print(logits)

# import py_vncorenlp

# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/zuanki/Project/LawDocRetriever/VnCoreNLP')
# text = "Giải thích từ ngữ\n\nTrong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:\n\n1. Người lao động là người làm việc cho người sử dụng lao động theo thỏa thuận, được trả lương và chịu sự quản lý, điều hành, giám sát của người sử dụng lao động.\n\nĐộ tuổi lao động tối thiểu của người lao động là đủ 15 tuổi, trừ trường hợp quy định tại Mục 1 Chương XI của Bộ luật này.\n\n2. Người sử dụng lao động là doanh nghiệp, cơ quan, tổ chức, hợp tác xã, hộ gia đình, cá nhân có thuê mướn, sử dụng người lao động làm việc cho mình theo thỏa thuận; trường hợp người sử dụng lao động là cá nhân thì phải có năng lực hành vi dân sự đầy đủ.\n\n3. Tổ chức đại diện người lao động tại cơ sở là tổ chức được thành lập trên cơ sở tự nguyện của người lao động tại một đơn vị sử dụng lao động nhằm mục đích bảo vệ quyền và lợi ích hợp pháp, chính đáng của người lao động trong quan hệ lao động thông qua thương lượng tập thể hoặc các hình thức khác theo quy định của pháp luật về lao động. Tổ chức đại diện người lao động tại cơ sở bao gồm công đoàn cơ sở và tổ chức của người lao động tại doanh nghiệp.\n\n4. Tổ chức đại diện người sử dụng lao động là tổ chức được thành lập hợp pháp, đại diện và bảo vệ quyền, lợi ích hợp pháp của người sử dụng lao động trong quan hệ lao động.\n\n5. Quan hệ lao động là quan hệ xã hội phát sinh trong việc thuê mướn, sử dụng lao động, trả lương giữa người lao động, người sử dụng lao động, các tổ chức đại diện của các bên, cơ quan nhà nước có thẩm quyền. Quan hệ lao động bao gồm quan hệ lao động cá nhân và quan hệ lao động tập thể.\n\n6. Người làm việc không có quan hệ lao động là người làm việc không trên cơ sở thuê mướn bằng hợp đồng lao động.\n\n7. Cưỡng bức lao động là việc dùng vũ lực, đe dọa dùng vũ lực hoặc các thủ đoạn khác để ép buộc người lao động phải làm việc trái ý muốn của họ.\n\n8. Phân biệt đối xử trong lao động là hành vi phân biệt, loại trừ hoặc ưu tiên dựa trên chủng tộc, màu da, nguồn gốc quốc gia hoặc nguồn gốc xã hội, dân tộc, giới tính, độ tuổi, tình trạng thai sản, tình trạng hôn nhân, tôn giáo, tín ngưỡng, chính kiến, khuyết tật, trách nhiệm gia đình hoặc trên cơ sở tình trạng nhiễm HIV hoặc vì lý do thành lập, gia nhập và hoạt động công đoàn, tổ chức của người lao động tại doanh nghiệp có tác động làm ảnh hưởng đến bình đẳng về cơ hội việc làm hoặc nghề nghiệp.\n\nViệc phân biệt, loại trừ hoặc ưu tiên xuất phát từ yêu cầu đặc thù của công việc và các hành vi duy trì, bảo vệ việc làm cho người lao động dễ bị tổn thương thì không bị xem là phân biệt đối xử.\n\n9. Quấy rối tình dục tại nơi làm việc là hành vi có tính chất tình dục của bất kỳ người nào đối với người khác tại nơi làm việc mà không được người đó mong muốn hoặc chấp nhận. Nơi làm việc là bất kỳ nơi nào mà người lao động thực tế làm việc theo thỏa thuận hoặc phân công của người sử dụng lao động."
# output = rdrsegmenter.word_segment(text)
# print(output)

# from rank_bm25 import BM25Okapi

# corpus = [
#     "Hello there good man!",
#     "It is quite windy in London",
#     "How is the weather today?"
# ]

# tokenized_corpus = [doc.split(" ") for doc in corpus]

# bm25 = BM25Okapi(tokenized_corpus)

# query = "today windy London weather good"
# tokenized_query = query.split(" ")

# doc_scores = bm25.get_scores(tokenized_query)

# print(doc_scores)

# from src.models.lexical import QLDRetrieval, BM25Retrieval, TFIDFRetrieval
# import json

# with open('data/BM25/2023/law.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# bm25 = BM25Retrieval(data)
# qld = QLDRetrieval(data)
# tfidf = TFIDFRetrieval(data)

# query = "hợp_tác quốc_tế nghiên_cứu khoa_học chuyển_giao công_nghệ cung_cấp dịch_vụ kỹ_thuật_số để phát_triển điện_ảnh là một trong các chính_sách của nhà_nước về phát_triển điện_ảnh công_nghiệp điện_ảnh là đúng hay sai"
# document = "phạm_vi điều_chỉnh luật này quy_định về phòng_chống ma_tuý quản_lý người sử_dụng trái_phép chất ma_tuý cai_nghiện ma_tuý trách_nhiệm của cá_nhân gia_đình cơ_quan tổ_chức trong phòng_chống ma_tuý quản_lý_nhà_nước và hợp_tác quốc_tế về phòng_chống ma_tuý"

# print(bm25.retrieve(query)[0])
# print(qld.retrieve(query)[0])
# print(tfidf.retrieve(query)[0])

# print(bm25.score(query, document))
# print(qld.score(query, document))
# print(tfidf.score(query, document))