"""RAG Pipeline Architecture Specification (12 Steps Across 4 Phases).

Mapped from the canonical End-to-End RAG Architecture:

PHASE 1: INGESTION ("Garbage in, garbage out")
  - Step 1: Data Sources (Collect from PDFs, APIs, websites, transcripts)
  - Step 2: Document Loading (Ingest content from various sources)
  - Step 3: Meaningful Chunking (Split documents into semantic chunks)
  - Step 4: Metadata Extraction (Extract title, author, tags, relations, and more)
  * Failure Points to Avoid: Chunks built for speed, not meaning; Metadata skipped to save time.

PHASE 2: INDEXING ("Building the Brain")
  - Step 5: Embeddings (Convert chunks into vector embeddings that capture meaning)
  - Step 6: Vector Database (Store embeddings in vector DB for fast, intelligent retrieval)
  * Output: Indexed Knowledge Base.

PHASE 3: RETRIEVAL ("The Make-or-Break Stage")
  - Step 7: Query Rewriting (Expand intent, clarify, and rewrite for better match)
  - Step 8: Hybrid Search (Combine semantic dense vector and keyword lexical BM25 search)
  - Step 9: Reranking (Rerank results using a strong cross-encoder relevance model)
  * Failure Point to Avoid: Retrieval returning volume, not relevance.

PHASE 4: GENERATION & EVALUATION ("Close the Loop")
  - Step 10: Context Assembly (Select, order, and compress the best context for prompt)
  - Step 11: LLM Generation (LLM synthesizes grounded answer using provided context)
  - Step 12: Evaluation (Evaluate faithfulness, latency, cost, and citations)
  * Failure Point to Avoid: No evaluation after generation.
  * Final Output: Grounded Answer (Relevant, grounded, cited, fast, trustworthy).
"""
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel


class RagPhase(str, Enum):
    INGESTION = "1_Ingestion"
    INDEXING = "2_Indexing"
    RETRIEVAL = "3_Retrieval"
    GENERATION_AND_EVALUATION = "4_Generation_and_Evaluation"


class PipelineStepInfo(BaseModel):
    step_number: int
    name: str
    phase: RagPhase
    service_owner: str
    description: str
    failure_risk: str = ""


PIPELINE_STEPS: List[PipelineStepInfo] = [
    # Phase 1: Ingestion
    PipelineStepInfo(
        step_number=1,
        name="Data Sources",
        phase=RagPhase.INGESTION,
        service_owner="ingestion-service",
        description="Collect from PDFs, APIs, websites, and transcripts.",
    ),
    PipelineStepInfo(
        step_number=2,
        name="Document Loading",
        phase=RagPhase.INGESTION,
        service_owner="ingestion-service",
        description="Ingest and parse content from various sources.",
    ),
    PipelineStepInfo(
        step_number=3,
        name="Meaningful Chunking",
        phase=RagPhase.INGESTION,
        service_owner="ingestion-service",
        description="Split documents into semantic, coherent chunks.",
        failure_risk="Chunks built for speed, not meaning.",
    ),
    PipelineStepInfo(
        step_number=4,
        name="Metadata Extraction",
        phase=RagPhase.INGESTION,
        service_owner="ingestion-service",
        description="Extract title, author, tags, relations, and publication metadata.",
        failure_risk="Metadata skipped to save time.",
    ),

    # Phase 2: Indexing
    PipelineStepInfo(
        step_number=5,
        name="Embeddings",
        phase=RagPhase.INDEXING,
        service_owner="embedding-service",
        description="Convert chunks into dense vector embeddings that capture semantic meaning.",
    ),
    PipelineStepInfo(
        step_number=6,
        name="Vector Database",
        phase=RagPhase.INDEXING,
        service_owner="retrieval-service",
        description="Store embeddings in a vector database for fast, intelligent retrieval (Indexed Knowledge).",
    ),

    # Phase 3: Retrieval
    PipelineStepInfo(
        step_number=7,
        name="Query Rewriting",
        phase=RagPhase.RETRIEVAL,
        service_owner="retrieval-service",
        description="Expand intent, clarify, and rewrite query for better match.",
    ),
    PipelineStepInfo(
        step_number=8,
        name="Hybrid Search",
        phase=RagPhase.RETRIEVAL,
        service_owner="retrieval-service",
        description="Combine semantic (dense vector) and keyword (lexical BM25) search.",
    ),
    PipelineStepInfo(
        step_number=9,
        name="Reranking",
        phase=RagPhase.RETRIEVAL,
        service_owner="retrieval-service",
        description="Rerank results using a cross-encoder model for high-precision relevance.",
        failure_risk="Retrieval returning volume, not relevance.",
    ),

    # Phase 4: Generation & Evaluation
    PipelineStepInfo(
        step_number=10,
        name="Context Assembly",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        service_owner="generation-service",
        description="Select, order, and compress the best context for the prompt.",
    ),
    PipelineStepInfo(
        step_number=11,
        name="LLM Generation",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        service_owner="generation-service",
        description="LLM synthesizes grounded answer using provided context.",
    ),
    PipelineStepInfo(
        step_number=12,
        name="Evaluation",
        phase=RagPhase.GENERATION_AND_EVALUATION,
        service_owner="generation-service",
        description="Evaluate faithfulness, latency, cost, and citation grounding.",
        failure_risk="No evaluation after generation.",
    ),
]
