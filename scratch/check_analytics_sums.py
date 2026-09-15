import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()
cur.execute("SELECT SUM(article_count) FROM analytics_country_year WHERE country_code = '1' OR country_code = 'US';")
print("ANALYTICS_COUNTRY_YEAR SUM FOR US:", cur.fetchone(), flush=True)

cur.execute("SELECT country_code, SUM(article_count) FROM analytics_country_year GROUP BY country_code ORDER BY sum DESC LIMIT 10;")
print("TOP ANALYTICS_COUNTRY_YEAR:", cur.fetchall(), flush=True)
