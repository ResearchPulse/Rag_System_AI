import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import psycopg2
from app.core.config import get_settings

s = get_settings()
c = psycopg2.connect(host=s.POSTGRES_HOST, port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = c.cursor()
cur.execute("""
    SELECT a.article_id, a.title, a.publication_year, a.citation_count, t.display_name
    FROM "Article" a
    LEFT JOIN "Topic" t ON a.primary_topic = t.topic_id
    WHERE a.citation_count >= 20 AND a.abstract IS NOT NULL AND length(a.abstract) > 100
    ORDER BY a.citation_count DESC
    LIMIT 25;
""")
print("\nRecent 2024-2026 articles:")
cur.execute("""
    SELECT a.article_id, a.title, a.publication_year, a.citation_count, t.display_name
    FROM "Article" a
    LEFT JOIN "Topic" t ON a.primary_topic = t.topic_id
    WHERE a.publication_year >= 2024 AND a.citation_count >= 15 AND a.abstract IS NOT NULL
    ORDER BY a.citation_count DESC
    LIMIT 15;
""")
for r in cur.fetchall():
    print(f"doc_{r[0]} | Year: {r[2]} | Citations: {r[3]:<3} | Topic: {r[4]} | Title: {r[1]}")
