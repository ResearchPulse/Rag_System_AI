import sys, io, os
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request, json

url = "http://127.0.0.1:8001/api/v1/chat"
queries = [
    "số lượng tác giả trong project này là bao nhiêu",
    "tổng số lượng tác giả và tổng số lượng bài báo trong project này là bao nhiêu ?"
]

for q in queries:
    payload = {
        "query": q,
        "project_id": 18,
        "user_id": "9377a17b-e4ce-4ce2-84d8-fdb2fb909ad3",
        "save_history": False
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    print("=" * 60, flush=True)
    print('USER QUERY:', q, flush=True)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print('ANSWER:\n', data.get('answer'), flush=True)
    except Exception as e:
        print('ERROR:', e, flush=True)
