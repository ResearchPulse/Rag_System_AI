import sys
sys.path.insert(0, ".")
import time
import psycopg2
from neo4j import GraphDatabase
from app.core.config import get_settings

def normalize_and_sync_neo4j():
    settings = get_settings()
    
    print(f"1. Connecting to PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}...")
    p_conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    p_cur = p_conn.cursor()

    print(f"2. Connecting to Neo4j: {settings.NEO4J_URI}...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    t_overall = time.perf_counter()

    # Step A: Clean slate in Neo4j to eliminate mixed-type duplicates
    print("\n--- Step A: Resetting Neo4j database (clearing legacy mixed nodes) ---")
    t0 = time.perf_counter()
    with driver.session() as session:
        # Delete all existing nodes and relationships in safe batches
        session.run("CALL { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 20000 ROWS;")
    print(f"Database cleared in {round(time.perf_counter() - t0, 2)}s")

    # Step B: Ensure constraints for blazing fast lookups & uniqueness
    print("\n--- Step B: Creating/verifying unique ID constraints ---")
    with driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;",
            "CREATE CONSTRAINT article_id_unique IF NOT EXISTS FOR (art:Article) REQUIRE art.id IS UNIQUE;",
            "CREATE CONSTRAINT journal_id_unique IF NOT EXISTS FOR (j:Journal) REQUIRE j.id IS UNIQUE;",
            "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE;",
            "CREATE CONSTRAINT keyword_id_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE;",
        ]
        for c in constraints:
            try:
                session.run(c)
            except Exception:
                pass

    # Step C: Import Authors (exact Integer IDs matching PostgreSQL)
    print("\n--- Step C: Importing Authors (110,560 rows) ---")
    p_cur.execute('SELECT author_id, display_name FROM "Author";')
    authors = p_cur.fetchall()
    total_authors = len(authors)
    batch_size = 10000
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, total_authors, batch_size):
            chunk = authors[i:i + batch_size]
            batch_data = [{"id": int(r[0]), "name": r[1] or ""} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                CREATE (a:Author {id: row.id, name: row.name})
                """,
                batch=batch_data
            )
            print(f"  -> Imported {min(i + batch_size, total_authors):,}/{total_authors:,} Authors", end="\r")
    print(f"\n  Done {total_authors:,} Authors in {round(time.perf_counter() - t0, 2)}s")

    # Step D: Import Articles (40,451 rows)
    print("\n--- Step D: Importing Articles (40,451 rows) ---")
    p_cur.execute('SELECT article_id, title, publication_year, doi, citation_count FROM "Article";')
    articles = p_cur.fetchall()
    total_articles = len(articles)
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, total_articles, batch_size):
            chunk = articles[i:i + batch_size]
            batch_data = [
                {
                    "id": int(r[0]),
                    "title": r[1] or "",
                    "publication_year": int(r[2]) if r[2] is not None else 2024,
                    "doi": r[3] or "",
                    "citation_count": int(r[4]) if r[4] is not None else 0
                }
                for r in chunk
            ]
            session.run(
                """
                UNWIND $batch AS row
                CREATE (art:Article {
                    id: row.id,
                    title: row.title,
                    publication_year: row.publication_year,
                    doi: row.doi,
                    citation_count: row.citation_count
                })
                """,
                batch=batch_data
            )
            print(f"  -> Imported {min(i + batch_size, total_articles):,}/{total_articles:,} Articles", end="\r")
    print(f"\n  Done {total_articles:,} Articles in {round(time.perf_counter() - t0, 2)}s")

    # Step E: Import Journals (3,041 rows)
    print("\n--- Step E: Importing Journals (3,041 rows) ---")
    p_cur.execute('SELECT journal_id, display_name, issn, country FROM "Journal";')
    journals = p_cur.fetchall()
    t0 = time.perf_counter()
    with driver.session() as session:
        batch_data = [
            {"id": int(r[0]), "name": r[1] or "", "issn": r[2] or "", "country": r[3] or ""}
            for r in journals
        ]
        session.run(
            """
            UNWIND $batch AS row
            CREATE (j:Journal {id: row.id, name: row.name, issn: row.issn, country: row.country})
            """,
            batch=batch_data
        )
    print(f"  Done {len(journals):,} Journals in {round(time.perf_counter() - t0, 2)}s")

    # Step F: Import Topics (5,408 rows)
    print("\n--- Step F: Importing Topics (5,408 rows) ---")
    p_cur.execute('SELECT topic_id, display_name FROM "Topic";')
    topics = p_cur.fetchall()
    t0 = time.perf_counter()
    with driver.session() as session:
        batch_data = [{"id": int(r[0]), "name": r[1] or ""} for r in topics]
        session.run(
            """
            UNWIND $batch AS row
            CREATE (t:Topic {id: row.id, name: row.name})
            """,
            batch=batch_data
        )
    print(f"  Done {len(topics):,} Topics in {round(time.perf_counter() - t0, 2)}s")

    # Step G: Import Keywords (21,065 rows)
    print("\n--- Step G: Importing Keywords (21,065 rows) ---")
    p_cur.execute('SELECT keyword_id, display_name FROM "Keyword";')
    keywords = p_cur.fetchall()
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(keywords), batch_size):
            chunk = keywords[i:i + batch_size]
            batch_data = [{"id": int(r[0]), "name": r[1] or ""} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                CREATE (k:Keyword {id: row.id, name: row.name})
                """,
                batch=batch_data
            )
    print(f"  Done {len(keywords):,} Keywords in {round(time.perf_counter() - t0, 2)}s")

    # Step H: Import Author-Article Relationships (WRITES) (138,156 rows)
    print("\n--- Step H: Creating WRITES relationships (138,156 rows) ---")
    p_cur.execute('SELECT author_id, article_id FROM "Author_Article";')
    writes_data = p_cur.fetchall()
    total_writes = len(writes_data)
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, total_writes, batch_size):
            chunk = writes_data[i:i + batch_size]
            batch_data = [{"author_id": int(r[0]), "article_id": int(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (a:Author {id: row.author_id})
                MATCH (art:Article {id: row.article_id})
                MERGE (a)-[:WRITES]->(art)
                """,
                batch=batch_data
            )
            print(f"  -> Created {min(i + batch_size, total_writes):,}/{total_writes:,} WRITES relations", end="\r")
    print(f"\n  Done WRITES relations in {round(time.perf_counter() - t0, 2)}s")

    # Step I: Generate COLLABORATES_WITH relationships
    print("\n--- Step I: Generating COLLABORATES_WITH relationships ---")
    t0 = time.perf_counter()
    with driver.session() as session:
        session.run(
            """
            MATCH (a1:Author)-[:WRITES]->(art:Article)<-[:WRITES]-(a2:Author)
            WHERE id(a1) < id(a2)
            MERGE (a1)-[r:COLLABORATES_WITH]-(a2)
            ON CREATE SET r.collaborations = 1
            ON MATCH SET r.collaborations = r.collaborations + 1;
            """
        )
    print(f"  Done COLLABORATES_WITH in {round(time.perf_counter() - t0, 2)}s")

    # Step J: Final Verification
    print("\n" + "=" * 60)
    print("  FINAL STANDARDIZED NEO4J METRICS:")
    print("=" * 60)
    with driver.session() as session:
        res = session.run("MATCH (n) RETURN distinct labels(n)[0] AS lbl, count(n) AS cnt ORDER BY cnt DESC;")
        for r in res:
            print(f"  {r['lbl']:<18}: {r['cnt']:,}")

        rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC;")
        print("\n  Relationships:")
        for r in rel_res:
            print(f"  {r['rel_type']:<22}: {r['cnt']:,}")

    total_time = round(time.perf_counter() - t_overall, 2)
    print("\n" + "=" * 60)
    print(f"  SYNCHRONIZATION & NORMALIZATION COMPLETE IN {total_time}s!")
    print("=" * 60)

    p_conn.close()
    driver.close()

if __name__ == "__main__":
    normalize_and_sync_neo4j()
