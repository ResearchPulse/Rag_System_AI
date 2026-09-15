import sys, os, io
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding='utf-8')

from app.modules.retrieval.query_classifier import QueryClassifier
from app.modules.retrieval.text_to_sql.engine import TextToSQLEngine

c = QueryClassifier()
engine = TextToSQLEngine()

queries = [
    "tác giả nào đang là tác giả có số lượng bài báo cao nhất trong project này ?",
    "tổng số lượng tác giả và tổng số lượng bài báo trong project này là bao nhiêu ?",
    "số lượng tác giả trong project này là bao nhiêu"
]

for q in queries:
    print("=" * 60, flush=True)
    print("QUERY:", q, flush=True)
    res = c.classify(q)
    print("CATEGORY:", res.category, flush=True)
    print("SUB_CATEGORY:", res.sub_category, flush=True)
    print("RECOMMENDED RETRIEVERS:", res.execution_plan.recommended_retrievers, flush=True)
    sql = engine.generate_sql(q, project_id=18)
    print("GENERATED SQL:", sql, flush=True)
    chunk = engine.execute_and_format(q, project_id=18)
    print("EXECUTED CHUNK CONTENT:\n", chunk.content, flush=True)
