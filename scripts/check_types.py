import sys
sys.path.insert(0, ".")
from neo4j import GraphDatabase
from app.core.config import get_settings

def check_types():
    s = get_settings()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    labels = ['Author', 'Article', 'Journal', 'Topic', 'Keyword']
    with driver.session() as session:
        for lbl in labels:
            int_cnt = session.run(f"MATCH (n:{lbl}) WHERE n.id IS :: INTEGER RETURN count(n) as c").single()['c']
            str_cnt = session.run(f"MATCH (n:{lbl}) WHERE n.id IS :: STRING RETURN count(n) as c").single()['c']
            other_cnt = session.run(f"MATCH (n:{lbl}) WHERE NOT (n.id IS :: INTEGER OR n.id IS :: STRING) RETURN count(n) as c").single()['c']
            print(f"{lbl:<10} | Integer id: {int_cnt:>10,} | String id: {str_cnt:>10,} | Other/None: {other_cnt:>10,}")

if __name__ == "__main__":
    check_types()
