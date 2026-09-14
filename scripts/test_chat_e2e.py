import urllib.request
import json

TEST_QUERIES = [
    "Xu hướng công bố bài báo khoa học về RAG trong năm 2025-2026 là gì?",
    "Có bao nhiêu bài báo về RAG trong năm 2024?",
    "Có bao nhiêu bài báo về Machine Learning năm 2024?",
    "Thống kê số lượng bài báo từ 2024 đến 2025",
    "Có bao nhiêu tác giả trong hệ thống?",
    "Tác giả Xue Qin Yu đã công bố những bài báo nào?",
    "Ai là người thường hợp tác với tác giả Xue Qin Yu?",
]

for q in TEST_QUERIES:
    print("\n" + "=" * 80)
    print(f"QUERY: {q}")
    payload = {
        "query": q,
        "top_k": 5,
        "include_contexts": True,
    }
    req = urllib.request.Request(
        "http://localhost:8000/api/v1/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("STATUS: 200 OK")
            print("LATENCY:", data.get("latency_breakdown"))
            print("CONTEXTS COUNT:", len(data.get("contexts") or []))
            print("ANSWER:\n" + data.get("answer"))
            if data.get("contexts"):
                print("\nSAMPLE CONTEXT 1 ID:", data["contexts"][0]["chunk_id"])
                print("SAMPLE CONTEXT 1 TITLE:", data["contexts"][0]["metadata"].get("title") or data["contexts"][0]["content"][:60])
    except Exception as e:
        print(f"FAILED: {e}")
