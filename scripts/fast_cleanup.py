import sys
sys.path.insert(0, ".")
import time
from neo4j import GraphDatabase
from app.core.config import get_settings

def clean_label(session, label: str, batch_size: int = 1000):
    print(f"\n>>> Cleaning legacy integer nodes for {label}...")
    total_deleted = 0
    t0 = time.perf_counter()
    
    while True:
        # 1. Fetch IDs of legacy nodes (ensure id is NOT null and is integer)
        res = session.run(f"MATCH (n:{label}) WHERE n.id IS NOT NULL AND n.id IS :: INTEGER RETURN n.id AS id LIMIT {batch_size}")
        ids = [r['id'] for r in res if r['id'] is not None]
        if not ids:
            break
            
        # 2. Delete using exact index lookup
        session.run(
            f"""
            UNWIND $ids AS target_id
            MATCH (n:{label} {{id: target_id}})
            DETACH DELETE n
            """,
            ids=ids
        )
        total_deleted += len(ids)
        print(f"  [Progress] Deleted {total_deleted:,} {label} legacy nodes...", end="\r", flush=True)
        
    print(f"\n  [Done] Finished {label}: removed {total_deleted:,} legacy nodes in {round(time.perf_counter() - t0, 2)}s")

def main():
    s = get_settings()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    
    t_start = time.perf_counter()
    with driver.session() as session:
        # First delete legacy nodes for all entities
        for lbl in ['Journal', 'Topic', 'Keyword', 'Article', 'Author']:
            clean_label(session, lbl, batch_size=1000)
            
    print(f"\nAll legacy nodes cleaned in {round(time.perf_counter() - t_start, 2)}s!")

if __name__ == "__main__":
    main()
