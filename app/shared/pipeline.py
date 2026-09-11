"""Canonical RAG Pipeline: 12 Steps across 4 Phases in the Modular Monolith."""
from enum import Enum
from typing import List
from pydantic import BaseModel


class RagPhase(str, Enum):
    INGESTION = "Phase 1: Ingestion"
    INDEXING = "Phase 2: Indexing"
    RETRIEVAL = "Phase 3: Retrieval"
    GENERATION_AND_EVALUATION = "Phase 4: Generation & Evaluation"


class PipelineStepInfo(BaseModel):
    step_number: int
    name: str
    phase: RagPhase
    module_path: str
    description: str
    failure_risk: str = ""


PIPELINE_STEPS: List[PipelineStepInfo] = [
    # Phase 1: Ingestion ("Garbage in, garbage out")
    PipelineStepInfo(
        step_number=1,
        name="Data Sources",
        phase=RagPhase.INGESTION,
        module_path="app.modules.ingestion.data_sources",
        description="Collect from PDFs, APIs, websites, and transcripts.",
    ),
    PipelineStepInfo(
        step_number=2,
        name="Document Loading",
        phase=RagPhase.INGESTION,
        module_path="app.modules.ingestion.loaders",
        description="Ingest and parse content from diverse scholarly documents.",
    ),
    PipelineStepInfo(
        step_number=3,
        name="Meaningful Chunking",
        phase=RagPhase.INGESTION,
        module_path="app.modules.ingestion.chunking",
        description="Split documents into semantic chunks preserving conceptual coherence.",
        failure_risk="Chunks built for speed, not meaning.",
    ),
    PipelineStepInfo(
        step_number=4,
        name="Metadata Extraction",
        phase=RagPhase.INGESTION,
        module_path="app.modules.ingestion.metadata",
        description="Extract title, authors, year, journal quartile, and cross-references.",
        failure_risk="Metadata skipped to save time.",
    ),

    # Phase 2: Indexing ("Building the Brain")
    PipelineStepInfo(
        step_number=5,
        name="Embeddings",
        phase=RagPhase.INDEXING,
        module_path="app.modules.indexing.embedders",
        description="Convert chunks into dense vector embeddings capturing deep semantics.",
    ),
    PipelineStepInfo(
        step_number=6,
        name="Vector Database",
        phase=RagPhase.INDEXING,
        module_path="app.modules.indexing.vector_store",
        description="Store embeddings in a high-speed vector DB for fast intelligent retrieval (Indexed Knowledge).",
    ),

    # Phase 3: Retrieval ("The Make-or-Break Stage")
    PipelineStepInfo(
        step_number=7,
        name="Query Rewriting",
        phase=RagPhase.RETRIEVAL,
        module_path="app.modules.retrieval.query_rewriting",
        description="Expand intent, clarify query semantics, and generate sub-queries.",
    ),
    PipelineStepInfo(
        step_number=8,
        name="Hybrid Search",
        phase=RagPhase.RETRIEVAL,
        module_path="app.modules.retrieval.hybrid_search",
        description="Fuse dense semantic search and sparse lexical (BM25) search.",
    ),
    PipelineStepInfo(
        step_number=9,
        name="Reranking",
        phase=RagPhase.RETRIEVAL,
        module_path="app.modules.retrieval.reranking",
        description="Score results with a cross-encoder model to prioritize precision over volume.",
        failure_risk="Retrieval returning volume, not relevance.",
    ),

    # Phase 4: Generation & Evaluation ("Close the Loop")
    PipelineStepInfo(
        step_number=10,
        name="Context Assembly",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        module_path="app.modules.generation.context_assembly",
        description="Select, order, deduplicate, and compress the best context for the LLM prompt.",
    ),
    PipelineStepInfo(
        step_number=11,
        name="LLM Generation",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        module_path="app.modules.generation.llm",
        description="LLM synthesizes grounded answer strictly conditioned on provided context.",
    ),
    PipelineStepInfo(
        step_number=12,
        name="Evaluation",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        module_path="app.modules.generation.evaluation",
        description="Evaluate faithfulness, latency, token cost, and citation attribution.",
        failure_risk="No evaluation after generation.",
    ),
]
