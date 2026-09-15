import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.retrieval.query_classifier import QueryClassifier

c = QueryClassifier()
res_c = c.classify("Mỹ có bao nhiêu bài báo ?")
print("=== CLASSIFIER ===")
print("CATEGORY:", res_c.category)
print("SUB_CATEGORY:", res_c.sub_category)
print("RETRIEVERS:", res_c.execution_plan.recommended_retrievers)

s = RetrievalService()
res = s.retrieve(RetrievalRequest(query="Mỹ có bao nhiêu bài báo ?", project_id=18))
print("\n=== RETRIEVAL RESULTS ===")
print("COUNT:", len(res.results))
for i, r in enumerate(res.results):
    print(f"--- Chunk {i} (type={r.metadata.get('type')}) ---")
    print(r.content)
