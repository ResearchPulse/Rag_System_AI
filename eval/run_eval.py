"""
Main Evaluation Suite Runner.
Loads tasks across regression, capability, and robustness tiers,
executes the RAG pipeline, runs graders, logs full transcripts (JSONL),
and prints a comprehensive pass-rate summary.
"""
import os
import sys
import glob
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any
import httpx

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


from eval.graders.retrieval_grader import RetrievalGrader
from eval.graders.generation_grader import ClaudeGenerationJudge
from eval.graders.robustness_grader import RobustnessGrader

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


class RagEvaluationRunner:
    def __init__(self):
        self.retrieval_grader = RetrievalGrader()
        self.generation_judge = ClaudeGenerationJudge()
        self.robustness_grader = RobustnessGrader()
        self.client = httpx.Client(timeout=60.0)

    def load_all_tasks(self) -> List[Dict[str, Any]]:
        task_files = glob.glob(os.path.join(BASE_DIR, "tasks", "**", "*.yaml"), recursive=True)
        tasks = []
        for tf in task_files:
            with open(tf, "r", encoding="utf-8") as f:
                docs = yaml.safe_load_all(f)
                for doc in docs:
                    if doc and isinstance(doc, dict) and "id" in doc:
                        tasks.append(doc)
        return tasks

    def run(self, filter_tier: str = None):
        all_tasks = self.load_all_tasks()
        if filter_tier:
            tasks = [t for t in all_tasks if t.get("tier") == filter_tier]
        else:
            tasks = all_tasks

        print(f"\n========================================================")
        print(f"🚀 RAG EVALUATION SUITE: {len(tasks)} tasks loaded (Filter: {filter_tier or 'ALL'})")
        print(f"📡 Target Endpoint: {RAG_API_URL}/chat")
        print(f"========================================================\n")

        stats = {
            "regression": {"total": 0, "passed": 0},
            "capability": {"total": 0, "passed": 0},
            "robustness": {"total": 0, "passed": 0},
            "by_grader": {
                "retrieval": {"total": 0, "passed": 0},
                "groundedness": {"total": 0, "passed": 0},
                "coverage": {"total": 0, "passed": 0},
                "language_relevance": {"total": 0, "passed": 0},
                "robustness": {"total": 0, "passed": 0},
            }
        }

        transcripts = []

        for idx, task in enumerate(tasks, start=1):
            task_id = task["id"]
            tier = task.get("tier", "capability")
            query = task["query"]
            expected_doc_ids = task.get("expected_doc_ids", [])
            ref_answer = task.get("reference_answer", "")
            grader_entries = task.get("graders", [])
            grader_configs = {
                (g if isinstance(g, str) else g["name"]): g
                for g in grader_entries
            }

            print(f"[{idx}/{len(tasks)}] [{tier.upper()}] Task {task_id}: {query[:60]}...")

            task_passed = True
            eval_results = {}
            retrieved_doc_ids = []
            contexts_text = []
            generated_answer = ""
            latency_ms = 0.0

            # 1. Call RAG End-to-End Chat API
            try:
                chat_res = self.client.post(
                    f"{RAG_API_URL}/chat",
                    json={"query": query, "top_k": 5, "include_contexts": True}
                )
                if chat_res.status_code != 200:
                    print(f"  ❌ Backend error {chat_res.status_code}: {chat_res.text[:100]}")
                    task_passed = False
                    eval_results["error"] = chat_res.text
                else:
                    data = chat_res.json()
                    generated_answer = data.get("answer", "")
                    raw_contexts = data.get("contexts", []) or []
                    contexts_text = [c.get("content", "") for c in raw_contexts]
                    retrieved_doc_ids = [
                        c.get("document_id") or c.get("metadata", {}).get("document_id") or c.get("chunk_id", "")
                        for c in raw_contexts
                    ]

                    latency_ms = sum(data.get("latency_breakdown", {}).values())
            except Exception as err:
                print(f"  ❌ Request exception: {err}")
                task_passed = False
                eval_results["exception"] = str(err)

            # 2. Execute Graders
            if generated_answer or retrieved_doc_ids:
                # A) Retrieval Grader
                if "retrieval" in grader_configs:
                    g_cfg = grader_configs["retrieval"]
                    thresholds = g_cfg.get("params", {}).get("thresholds") if isinstance(g_cfg, dict) else None
                    r_res = self.retrieval_grader.evaluate(
                        retrieved_doc_ids, expected_doc_ids, thresholds=thresholds
                    )
                    eval_results["retrieval"] = r_res
                    stats["by_grader"]["retrieval"]["total"] += 1
                    if r_res["passed"]:
                        stats["by_grader"]["retrieval"]["passed"] += 1
                    else:
                        task_passed = False

                # B) Groundedness Grader
                if "groundedness" in grader_configs:
                    gnd_res = self.generation_judge.grade_groundedness(query, contexts_text, generated_answer)
                    eval_results["groundedness"] = gnd_res
                    stats["by_grader"]["groundedness"]["total"] += 1
                    if gnd_res.get("verdict") == "PASS":
                        stats["by_grader"]["groundedness"]["passed"] += 1
                    elif gnd_res.get("verdict") == "FAIL":
                        task_passed = False

                # C) Coverage Grader
                if "coverage" in grader_configs:
                    cov_res = self.generation_judge.grade_coverage(query, ref_answer, generated_answer)
                    eval_results["coverage"] = cov_res
                    stats["by_grader"]["coverage"]["total"] += 1
                    if cov_res.get("verdict") == "PASS":
                        stats["by_grader"]["coverage"]["passed"] += 1
                    elif cov_res.get("verdict") == "FAIL":
                        task_passed = False

                # D) Language & Relevance Grader
                if "language_relevance" in grader_configs:
                    lang_res = self.generation_judge.grade_language_and_relevance(query, generated_answer)
                    eval_results["language_relevance"] = lang_res
                    stats["by_grader"]["language_relevance"]["total"] += 1
                    if lang_res.get("verdict") == "PASS":
                        stats["by_grader"]["language_relevance"]["passed"] += 1
                    else:
                        task_passed = False

                # E) Robustness Grader
                if "robustness" in grader_configs:
                    rob_res = self.robustness_grader.evaluate(generated_answer)
                    eval_results["robustness"] = rob_res
                    stats["by_grader"]["robustness"]["total"] += 1
                    if rob_res["passed"]:
                        stats["by_grader"]["robustness"]["passed"] += 1
                    else:
                        task_passed = False

            # Update tier totals
            if tier in stats:
                stats[tier]["total"] += 1
                if task_passed:
                    stats[tier]["passed"] += 1

            verdict_icon = "✅ PASS" if task_passed else "❌ FAIL"
            print(f"     └─ Result: {verdict_icon}")

            transcripts.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task_id,
                "tier": tier,
                "query": query,
                "expected_doc_ids": expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "generated_answer": generated_answer,
                "retrieved_contexts": contexts_text,
                "eval_results": eval_results,
                "task_passed": task_passed,
                "latency_ms": latency_ms
            })

        # 3. Save Artifacts
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = os.path.join(REPORT_DIR, f"transcript_{ts}.jsonl")
        latest_transcript = os.path.join(REPORT_DIR, "transcript_latest.jsonl")

        with open(transcript_file, "w", encoding="utf-8") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        with open(latest_transcript, "w", encoding="utf-8") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        self.print_and_save_summary(stats, transcript_file)

    def print_and_save_summary(self, stats: Dict[str, Any], transcript_path: str):
        summary_lines = [
            "\n" + "=" * 62,
            "📊 RAG PIPELINE EVALUATION SUMMARY REPORT",
            "=" * 62,
            "🎯 PASS RATE BY EVALUATION TIER:",
        ]

        for tier in ["regression", "capability", "robustness"]:
            t_info = stats[tier]
            tot = t_info["total"]
            pct = (t_info["passed"] / tot * 100) if tot > 0 else 0.0
            target_str = " (Target: ~100%)" if tier == "regression" else (" (Target: 60-80%)" if tier == "capability" else " (Target: ~100% rejection)")
            summary_lines.append(f"  • {tier.capitalize():<12}: {pct:>6.1f}%  ({t_info['passed']}/{tot}){target_str}")

        summary_lines.append("\n🔍 PASS RATE BY GRADER CRITERIA:")
        for g_name, g_info in stats["by_grader"].items():
            tot = g_info["total"]
            pct = (g_info["passed"] / tot * 100) if tot > 0 else 0.0
            summary_lines.append(f"  • {g_name:<20}: {pct:>6.1f}%  ({g_info['passed']}/{tot})")

        summary_lines.append("\n" + "-" * 62)
        summary_lines.append(f"📝 Full Transcripts Saved: {transcript_path}")
        summary_lines.append("=" * 62 + "\n")

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        summary_md_path = os.path.join(REPORT_DIR, "eval_summary_latest.md")
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(summary_text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite")
    parser.add_argument("--tier", choices=["regression", "capability", "robustness"], default=None, help="Filter by tier")
    args = parser.parse_args()

    runner = RagEvaluationRunner()
    runner.run(filter_tier=args.tier)
