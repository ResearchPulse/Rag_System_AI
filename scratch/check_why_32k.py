import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from app.modules.retrieval.text_to_sql.engine import TextToSQLEngine

import psycopg2
from app.core.config import get_settings
s = get_settings()

conn = None
for host in [s.POSTGRES_HOST, "100.121.61.95", "127.0.0.1"]:
    try:
        conn = psycopg2.connect(
            host=host,
            port=s.POSTGRES_PORT,
            dbname=s.POSTGRES_DB,
            user=s.POSTGRES_USER,
            password=s.POSTGRES_PASSWORD,
            connect_timeout=5
        )
        print(f"Connected successfully to {host}", flush=True)
        break
    except Exception as e:
        print(f"Failed {host}: {e}", flush=True)

if not conn:
    print("Could not connect to any host")
    sys.exit(1)

cur = conn.cursor()

cur.execute('''
    SELECT COUNT(DISTINCT a.article_id) 
    FROM "Article" a 
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.country = z.zone_id 
    WHERE (z.name = 'United States' OR z.code = 'US');
''')
print("TOTAL US PAPERS IN WHOLE SYSTEM:", cur.fetchone(), flush=True)

cur.execute('''
    SELECT COUNT(DISTINCT a.article_id) 
    FROM "Article" a 
    JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.country = z.zone_id 
    WHERE (z.name = 'United States' OR z.code = 'US')
      AND pas.project_id = 18;
''')
print("US PAPERS IN PROJECT 18:", cur.fetchone(), flush=True)

# Now check what SQL is generated for "Mỹ đang có bao nhiêu bài báo?" with project_id=18
q = "Mỹ đang có bao nhiêu bài báo?"
sql = engine.generate_sql(q, project_id=18)
print(f"\nGENERATED SQL FOR '{q}' with project_id=18:\n{sql}", flush=True)

validated = engine.validate_and_sanitize(sql, project_id=18)
print(f"SANITIZED SQL:\n{validated.sanitized_sql}", flush=True)

chunk = engine.execute_and_format(q, project_id=18)
print(f"\nCHUNK CONTENT:\n{chunk.content}", flush=True)
