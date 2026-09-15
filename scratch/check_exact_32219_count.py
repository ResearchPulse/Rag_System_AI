import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()

# Check analytics_country_year
cur.execute('SELECT * FROM "analytics_country_year" WHERE article_count = 32219 OR citation_count = 32219;')
print("analytics_country_year match:", cur.fetchall())

# Check analytics_journal_year
cur.execute('SELECT * FROM "analytics_journal_year" WHERE article_count = 32219 OR citation_count = 32219;')
print("analytics_journal_year match:", cur.fetchall())

# Check analytics_topic_year
cur.execute('SELECT * FROM "analytics_topic_year" WHERE article_count = 32219 OR citation_count = 32219;')
print("analytics_topic_year match:", cur.fetchall())

# Check analytics_keyword_year
cur.execute('SELECT * FROM "analytics_keyword_year" WHERE article_count = 32219 OR citation_count = 32219;')
print("analytics_keyword_year match:", cur.fetchall())
