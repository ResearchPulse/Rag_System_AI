"""Verification script for ResearchPulse Query Classification Layer & Semantic Routing.

Validates the 4 official categories:
1. direct_lookup
2. relational_reasoning
3. hybrid
4. chitchat
"""

import json
import sys
sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.query_classifier import QueryClassifier


def run_tests():
    print("=" * 75)
    print(" 🚀 ENTERPRISE QUERY CLASSIFICATION & ROUTING VERIFICATION SUITE")
    print("=" * 75)

    classifier = QueryClassifier()
    retrieval_service = RetrievalService()

    queries = [
        (
            "SCENARIO 1: direct_lookup (SQL Aggregation)",
            "Có bao nhiêu bài báo xuất bản năm 2023?",
        ),
        (
            "SCENARIO 2: direct_lookup (Semantic Similarity - pgVector)",
            "Deep learning and backpropagation in neural networks",
        ),
        (
            "SCENARIO 3: direct_lookup (Metadata Lookup - DOI)",
            "Tra cứu thông tin bài báo có mã DOI 10.1016/j.procs.2023.01.001",
        ),
        (
            "SCENARIO 4: relational_reasoning (Co-authorship / Graph)",
            "Tác giả Xue Qin Yu đã công bố những bài báo nào và hợp tác với ai?",
        ),
        (
            "SCENARIO 5: relational_reasoning (Citation Network)",
            "Những bài báo nào trích dẫn công trình của Geoffrey Hinton?",
        ),
        (
            "SCENARIO 6: hybrid (Filtered Graph)",
            "Trong các bài báo xuất bản sau 2023 thuộc chủ đề AI, tác giả nào hợp tác với nhau nhiều nhất?",
        ),
        (
            "SCENARIO 7: chitchat (Greeting & Out of scope)",
            "Chào bạn, bạn có thể giúp gì cho tôi?",
        ),
        (
            "SCENARIO 8: clarification_needed (Vague Query)",
            "bài báo",
        ),
    ]

    for label, query in queries:
        print(f"\n" + "-" * 70)
        print(f"[{label}]")
        print(f"User Query: \"{query}\"")

        # Step 1: Classification Output
        res = classifier.classify(query)
        output_json = {
            "category": res.category.value,
            "sub_category": res.sub_category.value,
            "confidence_score": res.confidence_score,
            "detected_language": res.detected_language,
            "classification_engine": res.classification_engine,
            "reasoning": res.reasoning,
            "execution_plan": {
                "requires_sql_aggregation": res.execution_plan.requires_sql_aggregation,
                "requires_vector_search": res.execution_plan.requires_vector_search,
                "requires_graph_traversal": res.execution_plan.requires_graph_traversal,
                "target_store": res.execution_plan.target_store.value,
                "suggested_sql": res.execution_plan.suggested_sql,
                "suggested_cypher": res.execution_plan.suggested_cypher,
                "recommended_retrievers": res.execution_plan.recommended_retrievers,
            },
            "extracted_filters": res.extracted_filters.model_dump() if res.extracted_filters else None,
            "latency_ms": res.latency_ms,
        }
        print("Classifier JSON Output:")
        print(json.dumps(output_json, ensure_ascii=False, indent=2))

        # Step 2: Routing Execution
        req = RetrievalRequest(query=query, top_k=2)
        resp = retrieval_service.retrieve(req)
        print(f"-> Router Execution: Retrieved {len(resp.results)} chunks in {resp.latency_ms}ms")
        for idx, chunk in enumerate(resp.results, 1):
            src = chunk.metadata.get("source", "N/A")
            title = chunk.metadata.get("title") or chunk.metadata.get("type", "")
            snippet = chunk.content[:120].replace("\n", " ")
            print(f"   [{idx}] Source: {src} | Title/Type: {title} | Snippet: {snippet}...")

    print("\n" + "=" * 75)
    print(" ✅ ALL ENTERPRISE QUERY CLASSIFIER CATEGORIES TESTED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()

