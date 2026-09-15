import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from app.modules.retrieval.text_to_sql.engine import TextToSQLEngine

engine = TextToSQLEngine()
conn = engine._get_postgres_connection()
if not conn:
    print("Could not connect to DB")
    sys.exit(1)

cur = conn.cursor()

# 1. Check Zone table for US / My / United States
cur.execute("""
    SELECT zone_id, name, code, type 
    FROM "Zone" 
    WHERE name ILIKE '%state%' OR name ILIKE '%mỹ%' OR name ILIKE '%america%' OR code IN ('US', 'USA');
""")
print("=== ZONE ROWS FOR US ===")
for r in cur.fetchall():
    print(r)

# 2. Check top countries in project 18 via Journal.country
cur.execute("""
    SELECT z.zone_id, z.name, z.code, COUNT(DISTINCT a.article_id) AS paper_count
    FROM "Article" a
    JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
    JOIN "Issue" i ON a.issue_id = i.issue_id
    JOIN "Volume" v ON i.volume_id = v.volume_id
    JOIN "Journal" j ON v.journal_id = j.journal_id
    JOIN "Zone" z ON j.country = z.zone_id
    WHERE pas.project_id = 18
    GROUP BY z.zone_id, z.name, z.code
    ORDER BY paper_count DESC
    LIMIT 10;
""")
print("\n=== TOP COUNTRIES IN PROJECT 18 VIA JOURNAL.COUNTRY ===")
for r in cur.fetchall():
    print(r)

# 3. What SQL does TextToSQL generate for "Mỹ có bao nhiêu bài báo ?"
print("\n=== GENERATED SQL FOR: 'Mỹ có bao nhiêu bài báo ?' ===")
sql = engine.generate_sql("Mỹ có bao nhiêu bài báo ?", project_id=18)
print("SQL:", sql)

chunk = engine.execute_and_format("Mỹ có bao nhiêu bài báo ?", project_id=18)
print("CHUNK CONTENT:\n", chunk.content)
