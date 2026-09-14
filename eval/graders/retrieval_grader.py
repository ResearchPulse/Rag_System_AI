"""
Code-based grader for Retrieval Service.
Calculates Precision@k, Recall@k, and Reciprocal Rank (RR).
"""
from typing import List, Dict, Any


class RetrievalGrader:
    def __init__(self):
        pass

    @staticmethod
    def calculate_metrics_at_k(
        retrieved_doc_ids: List[str],
        expected_doc_ids: List[str],
        k: int
    ) -> Dict[str, float]:
        """
        Calculates Precision@k and Recall@k.
        Formula:
          Precision@k = (Number of relevant docs in top-k) / k
          Recall@k    = (Number of relevant docs in top-k) / (Total relevant docs)
        """
        if not expected_doc_ids:
            return {f"precision@{k}": 1.0, f"recall@{k}": 1.0, f"hits@{k}": 0}

        top_k_retrieved = retrieved_doc_ids[:k]
        relevant_set = set(expected_doc_ids)
        retrieved_set = set(top_k_retrieved)
        hits = len(retrieved_set.intersection(relevant_set))

        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

        return {
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            f"hits@{k}": hits
        }

    @staticmethod
    def calculate_reciprocal_rank(
        retrieved_doc_ids: List[str],
        expected_doc_ids: List[str]
    ) -> float:
        """Calculates RR (1 / rank of the first relevant document retrieved)."""
        relevant_set = set(expected_doc_ids)
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_set:
                return round(1.0 / rank, 4)
        return 0.0

    def evaluate(
        self,
        retrieved_doc_ids: List[str],
        expected_doc_ids: List[str],
        k_list: List[int] = [3, 5],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval output against defined pass/fail thresholds.
        Default thresholds:
          recall@5 >= 0.8
          precision@3 >= 0.33
        """
        thresholds = thresholds or {"recall@5": 0.8, "precision@3": 0.33}
        metrics = {}
        for k in k_list:
            metrics.update(self.calculate_metrics_at_k(retrieved_doc_ids, expected_doc_ids, k))

        metrics["mrr"] = self.calculate_reciprocal_rank(retrieved_doc_ids, expected_doc_ids)

        passed = True
        failures = []
        for metric_name, min_val in thresholds.items():
            current_val = metrics.get(metric_name, 0.0)
            if current_val < min_val:
                passed = False
                failures.append(f"{metric_name}: {current_val} < threshold ({min_val})")

        return {
            "passed": passed,
            "metrics": metrics,
            "failures": failures,
            "retrieved_count": len(retrieved_doc_ids)
        }
