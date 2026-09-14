import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from neo4j import GraphDatabase
import psycopg2
from app.core.config import get_settings

s = get_settings()
p_conn = psycopg2.connect(
    host=s.POSTGRES_HOST,
    port=s.POSTGRES_PORT,
    dbname=s.POSTGRES_DB,
    user=s.POSTGRES_USER,
    password=s.POSTGRES_PASSWORD,
)
p_cur = p_conn.cursor()

d = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
with d.session() as session:
    print('Sample Topics:')
    for r in session.run('MATCH (t:Topic) RETURN t.name LIMIT 10'):
        print(' -', r['t.name'])
        
    print('\nSearch "y tế" in Topic:')
    res = list(session.run('MATCH (t:Topic) WHERE toLower(t.name) CONTAINS "y tế" RETURN t.name LIMIT 5'))
    print(res)
    
    print('\nSearch "y tế" in Keyword:')
    res_k = list(session.run('MATCH (k:Keyword) WHERE toLower(k.name) CONTAINS "y tế" RETURN k.name LIMIT 5'))
    print(res_k)
    
    print('\n--- Top authors in Health/Medicine in Neo4j ---')
    query = """
        MATCH (t:Topic)
        WHERE toLower(t.name) CONTAINS 'health' OR toLower(t.name) CONTAINS 'medic'
        MATCH (art:Article)-[:HAS_TOPIC]->(t)
        MATCH (a:Author)-[:WRITES]->(art)
        RETURN a.name AS author_name, count(DISTINCT art) AS paper_count
        ORDER BY paper_count DESC
        LIMIT 5
    """
    res = list(session.run(query))
    for r in res:
        print(f" - {r['author_name']}: {r['paper_count']} bài báo")
        
    print('\n--- Top authors in Health/Medicine in PostgreSQL ---')
    p_cur.execute("""
        SELECT a.display_name, COUNT(DISTINCT aa.article_id) AS paper_count
        FROM "Author" a
        JOIN "Author_Article" aa ON a.author_id = aa.author_id
        JOIN "Article" art ON aa.article_id = art.article_id
        JOIN "Topic" t ON art.primary_topic = t.topic_id
        WHERE t.display_name ILIKE '%health%' OR t.display_name ILIKE '%medic%'
        GROUP BY a.display_name
        ORDER BY paper_count DESC
        LIMIT 5;
    """)
    for r in p_cur.fetchall():
        print(f" - {r[0]}: {r[1]} bài báo")
