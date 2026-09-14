import sys
sys.path.insert(0, ".")
import time
import io
import psycopg2
from app.core.config import get_settings

def sync_all_tables():
    settings = get_settings()
    remote_url = "postgresql://postgres:postgres123@100.121.61.95:5432/researchpulse"
    local_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

    print(f"Connecting to remote server: {remote_url[:30]}...")
    s_conn = psycopg2.connect(remote_url)
    print(f"Connecting to local db: {local_url[:30]}...")
    d_conn = psycopg2.connect(local_url)

    s_cur = s_conn.cursor()
    d_cur = d_conn.cursor()

    # Get list of all tables in public schema
    s_cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
    tables = [t[0] for t in s_cur.fetchall() if not t[0].startswith("_")]

    # Priority order for core entities first, then relations, then analytics
    core_order = [
        "Subject_Area", "Subject_Category", "Topic", "Sub_Topic", "Keyword",
        "Publisher", "Journal", "Volume", "Issue", "Zone", "Ranking_Metric",
        "Journal_Ranking", "Journal_ISSN", "Journal_Subject_Category", "Journal_Ranking_Subject_Category",
        "Institution", "Author", "Institution_Author",
        # Article is already in local (40451), but references and mappings:
        "Author_Article", "Keyword_Article", "Article_Reference", "Article_Citing_Work",
        # Projects & Users
        "user", "wallet", "wallet_transaction", "coin_package", "payment_transaction",
        "Project", "Project_Member", "Project_Keyword", "Project_Article_Scope",
        "Project_Chat_Message", "Project_Article_Bookmark", "Bookmark",
        "Orcid_Scan_Job", "Orcid_Scan_Job_Item", "Password_Reset_Token",
        # Analytics tables
        "analytics_job", "analytics_country_year", "analytics_topic_year",
        "analytics_journal_year", "analytics_keyword_year",
        "rag_document_chunks", "system_log"
    ]

    # Add any other tables not in core_order
    for t in tables:
        if t not in core_order and t != "Article":
            core_order.append(t)

    print(f"\nDisabling foreign key constraints on local database for bulk copy...")
    d_cur.execute("SET session_replication_role = 'replica';")
    d_conn.commit()

    total_synced = 0
    t_start = time.perf_counter()

    for table in core_order:
        try:
            # Check remote count
            s_cur.execute(f'SELECT count(*) FROM "{table}";')
            r_count = s_cur.fetchone()[0]
            if r_count == 0:
                continue

            # Check local count
            d_cur.execute(f'SELECT count(*) FROM "{table}";')
            l_count = d_cur.fetchone()[0]

            if l_count >= r_count and table != "Article":
                print(f"[-] {table:<32} already up to date ({l_count:,} rows). Skipping.")
                continue

            print(f"[+] Syncing {table:<28} | Remote: {r_count:>8,} rows | Local: {l_count:>8,} rows ...", end="", flush=True)
            t0 = time.perf_counter()

            # Truncate local table if partial data exists
            if l_count > 0:
                d_cur.execute(f'TRUNCATE TABLE "{table}" CASCADE;')

            # Use binary or text copy stream via buffer
            # Streaming copy in chunks of 20,000 rows
            copy_query = f'COPY "{table}" TO STDOUT WITH (FORMAT binary);'
            copy_in = f'COPY "{table}" FROM STDIN WITH (FORMAT binary);'

            buf = io.BytesIO()
            s_cur.copy_expert(copy_query, buf)
            buf.seek(0)
            d_cur.copy_expert(copy_in, buf)
            d_conn.commit()

            elapsed = round(time.perf_counter() - t0, 2)
            print(f" DONE in {elapsed}s")
            total_synced += 1

        except Exception as e:
            d_conn.rollback()
            s_conn.rollback()
            print(f" ERROR: {e}")

    # Re-enable foreign key constraints
    print("\nRe-enabling foreign key constraints on local database...")
    d_cur.execute("SET session_replication_role = 'origin';")
    d_conn.commit()

    total_time = round(time.perf_counter() - t_start, 2)
    print(f"\n=======================================================")
    print(f"  SYNC COMPLETED! Synced {total_synced} tables in {total_time}s")
    print(f"=======================================================")

    s_conn.close()
    d_conn.close()

if __name__ == "__main__":
    sync_all_tables()
