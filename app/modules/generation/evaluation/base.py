"""Step 12: Evaluation - Evaluate faithfulness, latency, cost, and citation attribution."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class EvaluationReport(BaseModel):
    faithfulness_score: float  # Grounded in context (0.0 - 1.0)
    answer_relevance_score: float  # Addresses user query (0.0 - 1.0)
    citation_accuracy_score: float  # Correct citations (0.0 - 1.0)
    latency_ms: float
    estimated_cost_usd: float = 0.0
    passed_quality_gate: bool = True
    details: Dict[str, Any] = {}


class BaseRAGEvaluator(ABC):
    """Abstract interface for Step 12: Evaluation."""
    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        latency_ms: float,
    ) -> EvaluationReport:
        pass
