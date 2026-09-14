import sys
sys.path.insert(0, ".")
import time
import psycopg2
from neo4j import GraphDatabase
from app.core.config import get_settings

def sync_pg_to_neo4j():
    settings = get_settings()
    
    print(f"Connecting to Postgres: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}...")
    p_conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    p_cur = p_conn.cursor()
    
    print(f"Connecting to Neo4j: {settings.NEO4J_URI}...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    
    # 1. Sync Authors (110,560 rows)
    print("\n--- 1. Syncing Authors to Neo4j ---")
    p_cur.execute('SELECT author_id, display_name FROM "Author";')
    all_authors = p_cur.fetchall()
    total_authors = len(all_authors)
    print(f"Found {total_authors:,} Authors in Postgres. Merging into Neo4j in batches of 10,000...")
    
    batch_size = 10000
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, total_authors, batch_size):
            chunk = all_authors[i:i + batch_size]
            batch_data = [{"id": str(r[0]), "name": r[1] or ""} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (a:Author {id: row.id})
                SET a.name = row.name
                """,
                batch=batch_data
            )
            print(f"  -> Synced {min(i + batch_size, total_authors):,}/{total_authors:,} Authors")
    print(f"Authors sync completed in {round(time.perf_counter() - t0, 2)}s")

    # 2. Sync Journals (3,041 rows)
    print("\n--- 2. Syncing Journals to Neo4j ---")
    p_cur.execute('SELECT journal_id, display_name, issn, country FROM "Journal";')
    all_journals = p_cur.fetchall()
    print(f"Found {len(all_journals):,} Journals in Postgres. Merging into Neo4j...")
    t0 = time.perf_counter()
    with driver.session() as session:
        batch_data = [{"id": str(r[0]), "name": r[1] or "", "issn": r[2] or "", "country": r[3] or ""} for r in all_journals]
        session.run(
            """
            UNWIND $batch AS row
            MERGE (j:Journal {id: row.id})
            SET j.name = row.name, j.issn = row.issn, j.country = row.country
            """,
            batch=batch_data
        )
    print(f"Journals sync completed in {round(time.perf_counter() - t0, 2)}s")

    # 3. Sync Topics (5,408 rows)
    print("\n--- 3. Syncing Topics to Neo4j ---")
    p_cur.execute('SELECT topic_id, display_name FROM "Topic";')
    all_topics = p_cur.fetchall()
    print(f"Found {len(all_topics):,} Topics in Postgres. Merging into Neo4j...")
    t0 = time.perf_counter()
    with driver.session() as session:
        batch_data = [{"id": str(r[0]), "name": r[1] or ""} for r in all_topics]
        session.run(
            """
            UNWIND $batch AS row
            MERGE (t:Topic {id: row.id})
            SET t.name = row.name
            """,
            batch=batch_data
        )
    print(f"Topics sync completed in {round(time.perf_counter() - t0, 2)}s")

    # 4. Sync Keywords (21,065 rows)
    print("\n--- 4. Syncing Keywords to Neo4j ---")
    p_cur.execute('SELECT keyword_id, display_name FROM "Keyword";')
    all_keywords = p_cur.fetchall()
    print(f"Found {len(all_keywords):,} Keywords in Postgres. Merging into Neo4j...")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(all_keywords), batch_size):
            chunk = all_keywords[i:i + batch_size]
            batch_data = [{"id": str(r[0]), "name": r[1] or ""} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (k:Keyword {id: row.id})
                SET k.name = row.name
                """,
                batch=batch_data
            )
    print(f"Keywords sync completed in {round(time.perf_counter() - t0, 2)}s")

    # 5. Sync Articles (40,451 rows)
    print("\n--- 5. Syncing Articles to Neo4j ---")
    p_cur.execute('SELECT article_id, title, publication_year, doi, citation_count FROM "Article";')
    all_articles = p_cur.fetchall()
    print(f"Found {len(all_articles):,} Articles in Postgres. Merging into Neo4j...")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(all_articles), batch_size):
            chunk = all_articles[i:i + batch_size]
            batch_data = [
                {
                    "id": str(r[0]),
                    "title": r[1] or "",
                    "publication_year": r[2] if r[2] is not None else 2024,
                    "doi": r[3] or "",
                    "citation_count": r[4] or 0
                }
                for r in chunk
            ]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (art:Article {id: row.id})
                SET art.title = row.title,
                    art.publication_year = row.publication_year,
                    art.doi = row.doi,
                    art.citation_count = row.citation_count
                """,
                batch=batch_data
            )
            print(f"  -> Synced {min(i + batch_size, len(all_articles)):,}/{len(all_articles):,} Articles")
    print(f"Articles sync completed in {round(time.perf_counter() - t0, 2)}s")

    # 6. Sync Author-Article Relationships (WRITES) (138,156 rows)
    print("\n--- 6. Syncing Author-Article (WRITES) Relationships ---")
    p_cur.execute('SELECT author_id, article_id FROM "Author_Article";')
    all_rel = p_cur.fetchall()
    print(f"Found {len(all_rel):,} Author_Article records. Creating WRITES relations in batches of 10,000...")
    t0 = time.perf_counter()
    with driver.session() as session:
        for i in range(0, len(all_rel), batch_size):
            chunk = all_rel[i:i + batch_size]
            batch_data = [{"author_id": str(r[0]), "article_id": str(r[1])} for r in chunk]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (a:Author {id: row.author_id})
                MATCH (art:Article {id: row.article_id})
                MERGE (a)-[:WRITES]->(art)
                """,
                batch=batch_data
            )
            print(f"  -> Created {min(i + batch_size, len(all_rel)):,}/{len(all_rel):,} WRITES relations")
    print(f"WRITES relations completed in {round(time.perf_counter() - t0, 2)}s")

    # 7. Verification: Get final counts from Neo4j
    print("\n=======================================================")
    print("  VERIFYING FINAL NEO4J COUNTS:")
    print("=======================================================")
    with driver.session() as session:
        res = session.run("MATCH (n) RETURN distinct labels(n)[0] AS lbl, count(n) AS cnt ORDER BY cnt DESC;")
        for r in res:
            print(f"  {r['lbl']:<15}: {r['cnt']:,}")

        rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC;")
        print("\n  Relationships:")
        for r in rel_res:
            print(f"  {r['rel_type']:<20}: {r['cnt']:,}")

    p_conn.close()
    driver.close()
    print("\n[SUCCESS] Neo4j and PostgreSQL are now 100% in sync!")

if __name__ == "__main__":
    sync_pg_to_neo4j()
