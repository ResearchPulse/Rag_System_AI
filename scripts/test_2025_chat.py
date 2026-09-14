import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')

req = urllib.request.Request(
    'http://localhost:8000/api/v1/chat',
    data=json.dumps({
        'query': 'tìm danh sách các bài báo nổi bật trong lĩnh vực y tế trong năm 2025?',
        'top_k': 5,
        'model': 'gpt-4o-mini',
        'include_contexts': True,
        'temperature': 0.7
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print('ANSWER:\n', res.get('answer'))
    print('\nCONTEXTS:')
    for c in res.get('contexts', []):
        meta = c.get('metadata', {})
        print(f"- [{meta.get('year')}] {meta.get('title')} (Trích dẫn: {meta.get('citations')})")
