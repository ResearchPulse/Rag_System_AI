import sys
sys.path.insert(0, ".")
import time
import psycopg2
from neo4j import GraphDatabase
from app.core.config import get_settings

def sync_relationships():
    s = get_settings()
    p_conn = psycopg2.connect(
        host=s.POSTGRES_HOST,
        port=s.POSTGRES_PORT,
        dbname=s.POSTGRES_DB,
        user=s.POSTGRES_USER,
        password=s.POSTGRES_PASSWORD,
    )
    p_cur = p_conn.cursor()
    driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))
    batch_size = 10000

    # 1. Sync Article -> Journal (PUBLISHED_IN)
    print("\n--- 1. Syncing Article -> Journal (PUBLISHED_IN) ---")
    p_cur.execute("""
        SELECT a.article_id, j.journal_id
        FROM "Article" a
        JOIN "Issue" i ON a.issue_id = i.issue_id
        JOIN "Volume" v ON i.volume_id = v.volume_id
        JOIN "Journal" j ON v.journal_id = j.journal_id;
    """)
    art_journal = p_cur.fetchall()
    print(f"Found {len(art_journal):,} Article-Journal links.")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(art_journal), batch_size):
            chunk = art_journal[i:i + batch_size]
            batch_data = [{"art_id": str(r[0]), "journal_id": str(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (art:Article {id: row.art_id})
                MATCH (j:Journal {id: row.journal_id})
                MERGE (art)-[:PUBLISHED_IN]->(j)
                """,
                batch=batch_data
            )
            print(f"  -> Linked {min(i + batch_size, len(art_journal)):,}/{len(art_journal):,} articles to journals", end="\r", flush=True)
    print(f"\n  Done in {round(time.perf_counter() - t0, 2)}s")

    # 2. Sync Article -> Topic (HAS_TOPIC)
    print("\n--- 2. Syncing Article -> Topic (HAS_TOPIC) ---")
    p_cur.execute('SELECT article_id, primary_topic FROM "Article" WHERE primary_topic IS NOT NULL;')
    art_topic = p_cur.fetchall()
    print(f"Found {len(art_topic):,} Article-Topic links.")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(art_topic), batch_size):
            chunk = art_topic[i:i + batch_size]
            batch_data = [{"art_id": str(r[0]), "topic_id": str(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (art:Article {id: row.art_id})
                MATCH (t:Topic {id: row.topic_id})
                MERGE (art)-[:HAS_TOPIC]->(t)
                """,
                batch=batch_data
            )
            print(f"  -> Linked {min(i + batch_size, len(art_topic)):,}/{len(art_topic):,} articles to topics", end="\r", flush=True)
    print(f"\n  Done in {round(time.perf_counter() - t0, 2)}s")

    # 3. Sync Article -> Keyword (HAS_KEYWORD)
    print("\n--- 3. Syncing Article -> Keyword (HAS_KEYWORD) ---")
    p_cur.execute('SELECT article_id, keyword_id FROM "Keyword_Article";')
    art_keyword = p_cur.fetchall()
    print(f"Found {len(art_keyword):,} Keyword_Article links.")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(art_keyword), batch_size):
            chunk = art_keyword[i:i + batch_size]
            batch_data = [{"art_id": str(r[0]), "kw_id": str(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (art:Article {id: row.art_id})
                MATCH (k:Keyword {id: row.kw_id})
                MERGE (art)-[:HAS_KEYWORD]->(k)
                """,
                batch=batch_data
            )
            print(f"  -> Linked {min(i + batch_size, len(art_keyword)):,}/{len(art_keyword):,} articles to keywords", end="\r", flush=True)
    print(f"\n  Done in {round(time.perf_counter() - t0, 2)}s")

    # 4. Sync Co-authorship (COLLABORATES_WITH)
    print("\n--- 4. Computing Co-authorship (COLLABORATES_WITH) ---")
    p_cur.execute("""
        SELECT a1.author_id, a2.author_id
        FROM "Author_Article" a1
        JOIN "Author_Article" a2 ON a1.article_id = a2.article_id AND a1.author_id < a2.author_id;
    """)
    coauthors = p_cur.fetchall()
    print(f"Found {len(coauthors):,} Co-author pairs.")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(coauthors), batch_size):
            chunk = coauthors[i:i + batch_size]
            batch_data = [{"a1": str(r[0]), "a2": str(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (a1:Author {id: row.a1})
                MATCH (a2:Author {id: row.a2})
                MERGE (a1)-[:COLLABORATES_WITH]-(a2)
                """,
                batch=batch_data
            )
            print(f"  -> Created {min(i + batch_size, len(coauthors)):,}/{len(coauthors):,} co-author links", end="\r", flush=True)
    print(f"\n  Done in {round(time.perf_counter() - t0, 2)}s")

    # 5. Summary
    with driver.session() as session:
        print("\n=== FINAL RELATIONSHIPS IN NEO4J ===")
        rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC")
        for r in rel_res:
            print(f"  {r['rel_type']:<20}: {r['cnt']:,}")

    p_conn.close()
    driver.close()

if __name__ == "__main__":
    sync_relationships()
