import sys
sys.path.insert(0, ".")
import psycopg2
from neo4j import GraphDatabase
from app.core.config import get_settings

def check():
    s = get_settings()
    p_conn = psycopg2.connect(
        host=s.POSTGRES_HOST,
        port=s.POSTGRES_PORT,
        dbname=s.POSTGRES_DB,
        user=s.POSTGRES_USER,
        password=s.POSTGRES_PASSWORD
    )
    p_cur = p_conn.cursor()
    
    print("--- POSTGRESQL COUNTS ---")
    tables = ['Author', 'Article', 'Journal', 'Topic', 'Keyword', 'Author_Article', 'Keyword_Article', 'Issue', 'Volume']
    for t in tables:
        try:
            p_cur.execute(f'SELECT count(*) FROM "{t}";')
            print(f"Postgres {t:<15}: {p_cur.fetchone()[0]:,}")
        except Exception as e:
            p_conn.rollback()
            print(f"Postgres {t:<15}: ERROR {e}")
            
    print("\n--- NEO4J NODE COUNTS ---")
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    with driver.session() as session:
        labels = ['Author', 'Article', 'Journal', 'Topic', 'Keyword']
        for lbl in labels:
            try:
                c = session.run(f"MATCH (n:{lbl}) RETURN count(n) as total").single()['total']
                int_c = session.run(f"MATCH (n:{lbl}) WHERE n.id IS :: INTEGER RETURN count(n) as c").single()['c']
                str_c = session.run(f"MATCH (n:{lbl}) WHERE n.id IS :: STRING RETURN count(n) as c").single()['c']
                null_c = session.run(f"MATCH (n:{lbl}) WHERE n.id IS NULL RETURN count(n) as c").single()['c']
                print(f"Neo4j {lbl:<10}: Total={c:<10,} (Integer={int_c:,}, String={str_c:,}, Null={null_c:,})")
            except Exception as e:
                print(f"Neo4j {lbl:<10}: ERROR {e}")

        print("\n--- NEO4J RELATIONSHIP COUNTS ---")
        rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC")
        for r in rel_res:
            print(f"Neo4j rel {r['rel_type']:<20}: {r['cnt']:,}")

if __name__ == "__main__":
    check()
