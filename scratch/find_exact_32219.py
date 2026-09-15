import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import psycopg2
from app.core.config import get_settings
s = get_settings()
conn = psycopg2.connect(host='100.121.61.95', port=s.POSTGRES_PORT, dbname=s.POSTGRES_DB, user=s.POSTGRES_USER, password=s.POSTGRES_PASSWORD)
cur = conn.cursor()

cur.execute("""
    SELECT table_name, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND data_type IN ('integer', 'bigint', 'numeric', 'text', 'character varying');
""")
cols = cur.fetchall()
print(f"Checking {len(cols)} columns across all tables...")

matches = []
for tbl, col, dtype in cols:
    try:
        if 'int' in dtype or 'numeric' in dtype:
            cur.execute(f'SELECT count(*) FROM "{tbl}" WHERE "{col}" = 32219;')
        else:
            cur.execute(f'SELECT count(*) FROM "{tbl}" WHERE "{col}" ILIKE \'%32219%\' OR "{col}" ILIKE \'%32.219%\';')
        cnt = cur.fetchone()[0]
        if cnt > 0:
            print(f"FOUND MATCH in {tbl}.{col}: {cnt} rows!", flush=True)
            matches.append((tbl, col))
    except Exception:
        conn.rollback()

print("ALL MATCHES:", matches, flush=True)
