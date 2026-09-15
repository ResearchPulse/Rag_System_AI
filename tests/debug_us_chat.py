import sys, io, os
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request, json

url = "http://127.0.0.1:8001/api/v1/chat"
payload = {
    "query": "Mỹ có bao nhiêu bài báo ?",
    "project_id": 18,
    "user_id": "caveman_test",
    "save_history": False
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("ANSWER:\n", data.get("answer"), flush=True)
except Exception as e:
    print("ERROR:", e, flush=True)
