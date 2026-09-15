import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()
cur.execute('SELECT message_id, project_id, role, content, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, created_at FROM "Project_Chat_Message" WHERE message_id >= 130 ORDER BY message_id;')
for r in cur.fetchall():
    print(r, flush=True)
