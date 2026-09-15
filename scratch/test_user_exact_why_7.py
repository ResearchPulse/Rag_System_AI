import sys, os, io
sys.path.insert(0, os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.generation.service import GenerationService
from app.modules.generation.schemas import GenerationRequest

s = RetrievalService()
g = GenerationService()

queries = [
    "tổng số lượng tác giả và tổng số lượng bài báo trong project này là bao nhiêu ?",
    "số lượng tác giả trong project này là bao nhiêu"
]

for q in queries:
    print("=" * 60, flush=True)
    print("QUERY:", q, flush=True)
    ret = s.retrieve(RetrievalRequest(query=q, project_id=18))
    print("RETRIEVAL RESULTS COUNT:", len(ret.results), flush=True)
    print("RETRIEVAL RESULT COUNT:", len(ret.results), flush=True)
    for idx, r in enumerate(ret.results):
        print(f"--- Chunk {idx} (type={r.metadata.get('type')}) ---", flush=True)
        print(r.content, flush=True)
    
    gen = g.generate(GenerationRequest(query=q, contexts=ret.results, project_id=18))
    print("\nGENERATION ANSWER:\n", gen.answer, flush=True)
