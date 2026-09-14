import sys
sys.path.insert(0, ".")
import psycopg2
from app.core.config import get_settings

def check():
    s = get_settings()
    remote_url = "postgresql://postgres:postgres123@100.121.61.95:5432/researchpulse"
    
    r_conn = psycopg2.connect(remote_url)
    l_conn = psycopg2.connect(
        host=s.POSTGRES_HOST,
        port=s.POSTGRES_PORT,
        dbname=s.POSTGRES_DB,
        user=s.POSTGRES_USER,
        password=s.POSTGRES_PASSWORD,
    )
    
    r_cur = r_conn.cursor()
    l_cur = l_conn.cursor()
    
    r_cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
    tables = [t[0] for t in r_cur.fetchall() if not t[0].startswith("_")]
    
    print(f"{'Table Name':<35} | {'Local Count':<12} | {'Remote Count':<12}")
    print("-" * 65)
    
    missing = []
    for t in tables:
        try:
            r_cur.execute(f'SELECT count(*) FROM "{t}";')
            rc = r_cur.fetchone()[0]
        except Exception:
            r_conn.rollback()
            continue
            
        try:
            l_cur.execute(f'SELECT count(*) FROM "{t}";')
            lc = l_cur.fetchone()[0]
        except Exception:
            l_conn.rollback()
            lc = -1
            
        if rc != lc:
            print(f"{t:<35} | {lc:<12} | {rc:<12}")
            missing.append((t, lc, rc))
            
    r_conn.close()
    l_conn.close()
    return missing

if __name__ == "__main__":
    check()
