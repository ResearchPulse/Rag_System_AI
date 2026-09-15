import sys, os, io
sys.path.insert(0, os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.generation.service import GenerationService
from app.modules.generation.schemas import RagPipelineRequest

from app.modules.generation.context_memory.service import ContextMemoryService

s = RetrievalService()
g = GenerationService()
m = ContextMemoryService()

queries = [
    "tác giả nào đang là tác giả có số lượng bài báo cao nhất trong project này ?",
    "tổng số lượng tác giả và tổng số lượng bài báo trong project này là bao nhiêu ?",
    "số lượng tác giả trong project này là bao nhiêu"
]

for q in queries:
    print("=" * 60)
    print("USER QUERY:", q)
    req = RagPipelineRequest(
        query=q,
        project_id=18,
        user_id="test_user_7",
        save_history=False
    )
    res = g.execute_rag_pipeline(req, retrieval_service=s, context_memory_service=m)
    print("ANSWER:\n", res.answer)
