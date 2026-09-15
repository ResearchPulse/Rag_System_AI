import sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.generation.service import GenerationService
from app.modules.generation.schemas import GenerationRequest, ContextItem

s = RetrievalService()
g = GenerationService()

q = "Mỹ đang có bao nhiêu bài báo?"
ret = s.retrieve(RetrievalRequest(query=q, project_id=18))
print("RETRIEVAL CHUNKS:")
for idx, r in enumerate(ret.results):
    print(f"[{idx}] (type={r.metadata.get('type')})")
    print(r.content)

contexts_for_gen = [
    ContextItem(
        id=c.chunk_id,
        content=c.content,
        source=c.metadata.get("title", c.document_id),
        metadata=c.metadata,
    )
    for c in ret.results
]

# Generate with llama3.2:3b
res = g.generate(GenerationRequest(query=q, contexts=contexts_for_gen, project_id=18))
print("\n=== GENERATED ANSWER ===")
print(res.answer)
