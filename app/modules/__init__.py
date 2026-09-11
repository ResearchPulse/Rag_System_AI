"""Modular Monolith Core Domains for Rag_System_AI.

Phase 1: Ingestion  (Steps 1-4)
Phase 2: Indexing   (Steps 5-6)
Phase 3: Retrieval  (Steps 7-9)
Phase 4: Generation (Steps 10-12)
"""
from app.modules.generation.service import GenerationService
from app.modules.indexing.service import IndexingService
from app.modules.ingestion.service import IngestionService
from app.modules.retrieval.service import RetrievalService

__all__ = [
    "GenerationService",
    "IndexingService",
    "IngestionService",
    "RetrievalService",
]
