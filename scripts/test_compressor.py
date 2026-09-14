import sys
sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from app.modules.retrieval.query_rewriting.compressor import QueryCompressor

test_cases = [
    "Bạn ơi cho tôi hỏi là trong hệ thống có bao nhiêu bài báo về AI năm 2024 vậy ạ?",
    "Làm ơn cho mình biết tác giả Nguyễn Văn A đã hợp tác với ai thế nhỉ?",
    "Can you please tell me about deep learning research in the database?",
    "Xin chào bot, hãy giúp tôi thống kê số lượng bài báo của tạp chí IEEE nhé!",
    "Xu hướng nghiên cứu Graph Neural Network gần đây với nha",
    "Có bao nhiêu bài báo năm 2024?",
]

for q in test_cases:
    c = QueryCompressor.compress_query(q)
    reduction = round((1 - len(c) / len(q)) * 100, 1)
    print(f"Gốc: '{q}'\nNén: '{c}' (Giảm {reduction}% độ dài ký tự)\n")
