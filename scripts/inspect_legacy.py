import sys
sys.path.insert(0, ".")
from neo4j import GraphDatabase
from app.core.config import get_settings

def inspect_legacy():
    s = get_settings()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    labels = ['Author', 'Article', 'Journal', 'Topic', 'Keyword']
    with driver.session() as session:
        for lbl in labels:
            res = session.run(f"""
                MATCH (n:{lbl}) WHERE n.id IS NOT NULL AND n.id IS :: INTEGER
                RETURN count(n) AS cnt
            """).single()['cnt']
            
            # Check rels
            rel_res = session.run(f"""
                MATCH (n:{lbl})-[r]-() WHERE n.id IS NOT NULL AND n.id IS :: INTEGER
                RETURN type(r) AS rel_type, count(r) AS cnt
            """)
            rels = [f"{r['rel_type']}: {r['cnt']}" for r in rel_res]
            print(f"{lbl:<10} | Legacy int nodes: {res:>8,} | Rels: {', '.join(rels) if rels else 'None'}")

if __name__ == "__main__":
    inspect_legacy()
