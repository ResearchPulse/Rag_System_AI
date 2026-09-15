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

# Search for 32219 or 32,219 in any table
# Let's check articles count, zone, analytics tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public';
""")
tables = [r[0] for r in cur.fetchall()]
print("TABLES:", tables)

# Check analytics tables
for t in tables:
    if 'analytic' in t or 'trend' in t or 'stat' in t or 'count' in t or 'metric' in t or 'summary' in t:
        print(f"Checking table {t}...")
        try:
            cur.execute(f'SELECT * FROM "{t}" LIMIT 5;')
            print(f"Rows from {t}:", cur.fetchall())
        except Exception as e:
            conn.rollback()
            print(f"Error reading {t}: {e}")

# Also check how many articles in total in Article table
cur.execute('SELECT count(*) FROM "Article";')
print("TOTAL ARTICLES IN DB:", cur.fetchone())

# Check how many articles for each zone or country in the whole DB
cur.execute('''
    SELECT z.name, z.code, COUNT(DISTINCT a.article_id) 
    FROM "Article" a 
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.country = z.zone_id 
    GROUP BY z.name, z.code
    ORDER BY count DESC 
    LIMIT 10;
''')
print("TOP COUNTRIES IN ENTIRE DB:")
for r in cur.fetchall():
    print(r)
