import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
from app.modules.chat_history.repository import ChatHistoryRepository

repo = ChatHistoryRepository()
conn = repo.get_connection()
if conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT message_id, project_id, role, content, created_at 
            FROM "Project_Chat_Message" 
            ORDER BY message_id DESC 
            LIMIT 10;
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"ID: {r[0]} | Project: {r[1]} | Role: {r[2]}")
            print(f"Content: {r[3]}")
            print("-" * 50)
else:
    print("Could not connect to DB")
