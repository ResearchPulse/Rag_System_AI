"""
Pytest integration test for Retrieval Quality.
Asserts that Precision@k, Recall@k, and MRR meet the configured thresholds.
Usage:
  poetry run pytest eval/test_retrieval_pytest.py -v
  python eval/test_retrieval_pytest.py
"""
import os
import sys
import glob
import yaml

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from eval.graders.retrieval_grader import RetrievalGrader
except ImportError:
    from graders.retrieval_grader import RetrievalGrader

# Direct service execution for fast, deterministic, lifespan-safe testing
try:
    from app.modules.retrieval.service import RetrievalService
    from app.modules.retrieval.schemas import RetrievalRequest
    retrieval_service = RetrievalService()
except Exception:
    retrieval_service = None

test_client = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    import pytest
except ImportError:
    pytest = None

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1")
grader = RetrievalGrader()


def load_retrieval_tasks():
    tasks = []
    task_files = glob.glob(os.path.join(BASE_DIR, "tasks", "**", "*.yaml"), recursive=True)
    for tf in task_files:
        with open(tf, "r", encoding="utf-8") as f:
            docs = yaml.safe_load_all(f)
            for doc in docs:
                if not doc or not isinstance(doc, dict):
                    continue
                graders = [g if isinstance(g, str) else g.get("name") for g in doc.get("graders", [])]
                if "retrieval" in graders and doc.get("expected_doc_ids"):
                    tasks.append(doc)
    return tasks


RETRIEVAL_TASKS = load_retrieval_tasks()


def query_retrieval_endpoint(query: str, top_k: int = 5) -> list:
    """Queries retrieval service directly or via HTTP."""
    # 1. Direct Python service invocation (fastest, no network/lifespan overhead)
    if retrieval_service:
        try:
            res = retrieval_service.retrieve(RetrievalRequest(query=query, top_k=top_k))
            return [c.document_id for c in res.results]
        except Exception:
            pass

    payload = {"query": query, "top_k": top_k, "rerank": True}

    # 2. Fallback to live HTTP if server is running
    if httpx:
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(f"{RAG_API_URL}/retrieve", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return [item["document_id"] for item in data.get("results", [])]
        except Exception:
            pass

    return []


def run_single_task_evaluation(task: dict):
    query = task["query"]
    expected_doc_ids = task["expected_doc_ids"]

    # Extract thresholds
    thresholds = {"recall@5": 0.5, "precision@3": 0.2}
    for g in task.get("graders", []):
        if isinstance(g, dict) and g.get("name") == "retrieval":
            if "params" in g and "thresholds" in g["params"]:
                thresholds = g["params"]["thresholds"]

    retrieved_doc_ids = query_retrieval_endpoint(query, top_k=5)
    result = grader.evaluate(retrieved_doc_ids, expected_doc_ids, k_list=[3, 5], thresholds=thresholds)
    return result, retrieved_doc_ids


# Pytest test function (decorated conditionally if pytest is available)
if pytest:
    @pytest.mark.parametrize("task", RETRIEVAL_TASKS, ids=lambda t: f"{t.get('tier', 'test')}_{t.get('id', 'task')}")
    def test_retrieval_service_quality(task):
        result, retrieved_doc_ids = run_single_task_evaluation(task)
        assert result["passed"] is True, (
            f"Retrieval quality failed for task '{task['id']}'. "
            f"Retrieved: {retrieved_doc_ids}, Expected: {task.get('expected_doc_ids')}. "
            f"Failures: {result['failures']}"
        )


if __name__ == "__main__":
    print(f"Running Retrieval Quality Tests on {len(RETRIEVAL_TASKS)} tasks...")
    passed_count = 0
    for t in RETRIEVAL_TASKS:
        res, retrieved = run_single_task_evaluation(t)
        status = "PASS" if res["passed"] else "FAIL"
        if res["passed"]:
            passed_count += 1
            print(f"  [PASS] {t['id']}: {t['query'][:50]}")
        else:
            print(f"  [FAIL] {t['id']}: {t['query'][:50]}")
            print(f"         Retrieved: {retrieved} | Expected: {t.get('expected_doc_ids')}")
            print(f"         Failures: {res['failures']}")
    print(f"\nSummary: {passed_count}/{len(RETRIEVAL_TASKS)} passed ({(passed_count/len(RETRIEVAL_TASKS)*100):.1f}%)")
