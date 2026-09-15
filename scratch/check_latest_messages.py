import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()

cur.execute('SELECT message_id, project_id, role, content FROM "Project_Chat_Message" ORDER BY message_id DESC LIMIT 6;')
rows = cur.fetchall()
for r in rows:
    print(f"ID: {r[0]} | PID: {r[1]} | ROLE: {r[2]}")
    print(f"CONTENT: {r[3]}")
    print("-" * 50)
