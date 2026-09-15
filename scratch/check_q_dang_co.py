import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from app.modules.retrieval.query_classifier import QueryClassifier
c = QueryClassifier()
res = c.classify("Mỹ đang có bao nhiêu bài báo?")
print("CATEGORY:", res.category, flush=True)
print("SUB_CATEGORY:", res.sub_category, flush=True)
print("RECOMMENDED:", res.execution_plan.recommended_retrievers, flush=True)
print("EXTRACTED FILTERS:", res.extracted_filters, flush=True)
