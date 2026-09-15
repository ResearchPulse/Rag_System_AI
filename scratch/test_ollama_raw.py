import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request, json

ollama_url = "http://localhost:11434/api/generate"
prompts = [
    "Bạn là trợ lý AI học thuật ResearchPulse. Dựa vào tài liệu dưới đây, hãy trả lời súc tích, chính xác bằng tiếng Việt:\n\n--- TÀI LIỆU ---\n\n\n--- CÂU HỎI ---\nMỹ đang có bao nhiêu bài báo?\n\nTrả lời:",
    "Mỹ đang có bao nhiêu bài báo khoa học?",
    "How many scientific papers does the US have?"
]

for p in prompts:
    payload = {
        "model": "llama3.2:3b",
        "prompt": p,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    req = urllib.request.Request(ollama_url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("PROMPT:\n", p[:100], "...")
            print("RESPONSE:\n", data.get("response"))
            print("=" * 60)
    except Exception as e:
        print("ERROR:", e)
