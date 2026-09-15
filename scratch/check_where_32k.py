import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()

conn = psycopg2.connect(
    host="100.121.61.95",
    port=s.POSTGRES_PORT,
    dbname=s.POSTGRES_DB,
    user=s.POSTGRES_USER,
    password=s.POSTGRES_PASSWORD,
    connect_timeout=5
)
cur = conn.cursor()

# 1. Citations for US in whole system or project 18
cur.execute('''
    SELECT SUM(a.citation_count) 
    FROM "Article" a 
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.country = z.zone_id 
    WHERE (z.name = 'United States' OR z.code = 'US');
''')
print("SUM CITATIONS WHOLE SYSTEM US:", cur.fetchone(), flush=True)

# 2. Total citations project 18 US
cur.execute('''
    SELECT SUM(a.citation_count) 
    FROM "Article" a 
    JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.country = z.zone_id 
    WHERE (z.name = 'United States' OR z.code = 'US')
      AND pas.project_id = 18;
''')
print("SUM CITATIONS PROJECT 18 US:", cur.fetchone(), flush=True)

# 3. analytics_country_year for US
cur.execute('''
    SELECT SUM(article_count) FROM "analytics_country_year" WHERE country_code = 'US';
''')
print("analytics_country_year US:", cur.fetchone(), flush=True)

# 4. Now what query generates 32,219? Let's check TextToSQLEngine for "Mỹ đang có bao nhiêu bài báo?"
from app.modules.retrieval.text_to_sql.engine import TextToSQLEngine
engine = TextToSQLEngine()
q = "Mỹ đang có bao nhiêu bài báo?"
sql = engine.generate_sql(q, project_id=18)
print(f"\nGENERATED SQL FOR '{q}':\n{sql}", flush=True)
chunk = engine.execute_and_format(q, project_id=18)
print(f"CHUNK CONTENT:\n{chunk.content}", flush=True)
