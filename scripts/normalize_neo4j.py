import sys
sys.path.insert(0, ".")
import time
from neo4j import GraphDatabase
from app.core.config import get_settings

def delete_legacy_nodes(session, label: str):
    print(f"Cleaning legacy integer nodes for {label}...")
    deleted_total = 0
    t0 = time.perf_counter()
    while True:
        # Delete in small batches of 2000 to avoid Java heap space
        res = session.run(f"""
            MATCH (n:{label})
            WHERE n.id IS :: INTEGER
            WITH n LIMIT 2000
            DETACH DELETE n
            RETURN count(n) AS cnt
        """)
        cnt = res.single()['cnt']
        deleted_total += cnt
        if cnt == 0:
            break
        print(f"  Deleted {deleted_total:,} legacy {label} nodes...", end="\r")
    print(f"  Finished {label}: removed {deleted_total:,} legacy nodes in {round(time.perf_counter() - t0, 2)}s")

def run():
    s = get_settings()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    with driver.session() as session:
        for lbl in ['Author', 'Article', 'Journal', 'Topic', 'Keyword']:
            delete_legacy_nodes(session, lbl)

if __name__ == "__main__":
    run()
