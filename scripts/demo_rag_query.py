import json
import os
import sys
import urllib.request
import psycopg2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "researchpulse")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres123")


def embed_query(query: str, dimension: int = 768) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "content": {"parts": [{"text": query[:3000]}]},
        "outputDimensionality": dimension,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        return data["embedding"]["values"]


def search_articles(query_vector: list, top_k: int = 3):
    dsn = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    vector_str = f"[{','.join(map(str, query_vector))}]"
    sql = """
        SELECT article_id, title, COALESCE(abstract, 'No abstract'), publication_year,
               1 - (embedding <=> %s::vector) AS score
        FROM "Article"
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cur.execute(sql, (vector_str, vector_str, top_k))
    rows = cur.fetchall()
    conn.close()
    return rows


def generate_answer(query: str, articles: list) -> str:
    context_blocks = []
    for idx, row in enumerate(articles, 1):
        art_id, title, abstract, year, score = row
        context_blocks.append(
            f"[{idx}] Tiêu đề: {title} (Năm: {year}, Độ tương đồng: {round(score*100, 1)}%)\n"
            f"Nội dung tóm tắt: {abstract[:800]}\n"
        )
    context_text = "\n".join(context_blocks)

    system_prompt = (
        "Bạn là Chuyên gia Nghiên cứu Khoa học (Scientific Journal AI Assistant).\n"
        "Dựa vào các bài báo khoa học được trích xuất từ cơ sở dữ liệu dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác, học thuật, có dẫn chứng rõ ràng tên bài báo.\n\n"
        f"--- NGỮ CẢNH CÁC BÀI BÁO TÌM THẤY ---\n{context_text}\n"
        f"--- CÂU HỎI ---\n{query}\n\n"
        "Hãy tổng hợp câu trả lời bằng tiếng Việt và ghi chú nguồn bài báo tham khảo:"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": system_prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
        return data["candidates"][0]["content"]["parts"][0]["text"]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Các nghiên cứu về y học hoặc trí tuệ nhân tạo nổi bật"
    print(f"\n❓ CÂU HỎI: {query}")
    print("⏳ 1. Đang chuyển câu hỏi thành Vector bằng Gemini Embedding...")
    vec = embed_query(query)

    print("⏳ 2. Đang tìm kiếm bài báo tương đồng ngữ nghĩa trong PostgreSQL (pgvector)...")
    articles = search_articles(vec, top_k=3)

    print(f"🎯 Đã tìm thấy {len(articles)} bài báo khớp nhất trong database:")
    for idx, (aid, title, abstract, year, score) in enumerate(articles, 1):
        print(f"   [{idx}] (ID: {aid}, Sim: {score:.4f}, Năm: {year}) - {title[:80]}...")

    print("\n⏳ 3. Đang gửi ngữ cảnh sang Gemini 2.5 Flash để tổng hợp câu trả lời...")
    answer = generate_answer(query, articles)

    print("\n" + "="*70)
    print("🤖 CÂU TRẢ LỜI CỦA GEMINI AI:")
    print("="*70)
    print(answer)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
