import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import psycopg2
from app.core.config import get_settings

s = get_settings()
c = psycopg2.connect(host=s.POSTGRES_HOST, port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = c.cursor()
cur.execute('SELECT publication_year, count(*) FROM "Article" WHERE publication_year >= 2020 GROUP BY publication_year ORDER BY publication_year DESC;')
print('Articles by year >= 2020:')
for r in cur.fetchall():
    print(f' - Năm {r[0]}: {r[1]:,} bài')

cur.execute("""
    SELECT a.article_id, a.title, a.publication_year, a.citation_count
    FROM "Article" a
    WHERE (a.title ILIKE '%health%' OR a.title ILIKE '%medic%' OR a.abstract ILIKE '%health%' OR a.abstract ILIKE '%medic%')
      AND a.publication_year = 2025
    ORDER BY a.citation_count DESC
    LIMIT 5;
""")
print('\nTop 2025 articles with health/medic in title/abstract:')
for r in cur.fetchall():
    print(f" - [{r[2]}] (Citations: {r[3]}) {r[1]}")

cur.execute('SELECT MAX(publication_year), MIN(publication_year) FROM "Article";')
print('\nMin and Max publication year in database:', cur.fetchone())
