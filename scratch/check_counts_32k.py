import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()

# 1. Total articles without is_deleted
cur.execute('SELECT count(*) FROM "Article" WHERE is_deleted IS NOT TRUE;')
print("Active articles:", cur.fetchone()[0])

# 2. Total articles with DOI
cur.execute('SELECT count(*) FROM "Article" WHERE doi IS NOT NULL;')
print("Articles with DOI:", cur.fetchone()[0])

# 3. Articles by zone region: NA (Northern America)?
cur.execute('''
    SELECT count(DISTINCT a.article_id) 
    FROM "Article" a 
    JOIN "Issue" i ON a.issue_id = i.issue_id 
    JOIN "Volume" v ON i.volume_id = v.volume_id 
    JOIN "Journal" j ON v.journal_id = j.journal_id 
    JOIN "Zone" z ON j.region = z.zone_id 
    WHERE z.code = 'NA' OR z.name = 'Northern America';
''')
print("Articles in Northern America:", cur.fetchone()[0])

# 4. Total Author_Article links?
cur.execute('SELECT count(*) FROM "Author_Article";')
print("Author_Article links:", cur.fetchone()[0])

# 5. Articles with primary_topic IS NOT NULL?
cur.execute('SELECT count(*) FROM "Article" WHERE primary_topic IS NOT NULL;')
print("Articles with topic:", cur.fetchone()[0])

# 6. Articles in English?
cur.execute("SELECT count(*) FROM \"Article\" WHERE language = 'en';")
print("Articles in en:", cur.fetchone()[0])
