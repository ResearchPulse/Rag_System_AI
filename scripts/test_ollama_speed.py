import urllib.request
import json
import time

prompt = """Phân loại câu hỏi nghiên cứu khoa học:
Categories:
- "direct_lookup" (sub: "sql_aggregation" nếu đếm/thống kê, "semantic_similarity" nếu tìm nội dung)
- "relational_reasoning" (sub: "co_authorship" nếu đồng tác giả, "author_publications" nếu bài của tác giả)
- "hybrid" (sub: "topic_clustering" nếu hỏi xu hướng, "filtered_graph" nếu lọc + quan hệ)
- "chitchat" (sub: "chitchat")

Query: "Xu hướng nghiên cứu về Blockchain trong y tế năm 2024"

Output format JSON:
{"category": "hybrid", "sub_category": "topic_clustering", "extracted_filters": {"keyword": "Blockchain", "year": 2024}}"""

t0 = time.time()
req = urllib.request.Request(
    'http://localhost:11434/api/generate',
    data=json.dumps({
        'model': 'llama3.2:3b',
        'prompt': prompt,
        'format': 'json',
        'stream': False,
        'options': {'temperature': 0.0, 'num_predict': 100}
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req, timeout=30)
data = json.loads(res.read().decode())
print(f"Time taken: {time.time() - t0:.2f}s")
print(data['response'])
