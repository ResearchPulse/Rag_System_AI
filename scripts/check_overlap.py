import sys
sys.path.insert(0, ".")
from neo4j import GraphDatabase
from app.core.config import get_settings

def check_overlap():
    s = get_settings()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    labels = ['Author', 'Journal', 'Topic', 'Keyword', 'Article']
    with driver.session() as session:
        for lbl in labels:
            res = session.run(f"""
                MATCH (n1:{lbl}) WHERE n1.id IS :: INTEGER
                OPTIONAL MATCH (n2:{lbl}) WHERE n2.id = toString(n1.id)
                RETURN 
                    count(n1) AS total_int,
                    count(n2) AS matched_str,
                    count(CASE WHEN n2 IS NULL THEN 1 END) AS missing_str
            """).single()
            print(f"{lbl:<10}: int={res['total_int']:>6,}, matched_str={res['matched_str']:>6,}, missing_str={res['missing_str']:>6,}")

if __name__ == "__main__":
    check_overlap()
