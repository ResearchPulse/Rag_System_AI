import sys
sys.path.insert(0, ".")
import psycopg2
from app.core.config import get_settings

s = get_settings()
c = psycopg2.connect(host=s.POSTGRES_HOST, port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = c.cursor()

cur.execute("""
    SELECT count(*)
    FROM "Article" a
    JOIN "Issue" i ON a.issue_id = i.issue_id
    JOIN "Volume" v ON i.volume_id = v.volume_id
    JOIN "Journal" j ON v.journal_id = j.journal_id;
""")
print('Total Article to Journal links in PG:', cur.fetchone()[0])

cur.execute('SELECT count(*) FROM "Article" WHERE primary_topic IS NOT NULL;')
print('Articles with primary_topic in PG:', cur.fetchone()[0])

cur.execute('SELECT count(*) FROM "Keyword_Article";')
print('Keyword_Article in PG:', cur.fetchone()[0])
