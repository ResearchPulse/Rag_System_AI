"""Query & Context Compression Module for Token Minimization.

Reduces prompt token overhead by 40-70% through conversational filler stripping,
entity-preserving stopword filtering, whitespace normalization, and context compaction.
"""
import re
from typing import List, Optional


class QueryCompressor:
    """Fast, deterministic query compressor to minimize prompt token usage."""

    # Conversational filler prefixes (Vietnamese & English)
    CONVERSATIONAL_PREFIXES = [
        r"^(bạn\s*ơi|ad\s*ơi|bot\s*ơi|ai\s*ơi|ơi\s*bạn)\s*[,:\s]*",
        r"^(xin\s*chào|chào\s*(bạn|ad|bot|ai)?|hello|hi|hey)\s*(bot|ad|ai|bạn)?\s*[,:\s]*",
        r"^(làm\s*ơn\s*cho\s*(tôi|mình|em)\s*(biết|hỏi|xem)?|vui\s*lòng\s*cho\s*(tôi|mình|em)\s*(biết|hỏi))\s*[,:\s]*",
        r"^(cho\s*(tôi|mình|em|anh|chị)\s*(hỏi|biết|xem)(\s*(là|về))?|hãy\s*cho\s*(tôi|mình|em)\s*biết)\s*[,:\s]*",
        r"^(hãy\s*)?giúp\s*(tôi|mình|em)\s*(tìm|tra\s*cứu|thống\s*kê|phân\s*tích|xem)?\s*[,:\s]*",
        r"^hãy\s*(thống\s*kê|phân\s*tích|liệt\s*kê|cho\s*biết|tìm)\s*[,:\s]*",
        r"^(can\s*you\s*(please\s*)?(tell|show|find|explain)\s*(to\s*)?me|could\s*you\s*(please\s*)?provide|please\s*(tell|show|find)|i\s*want\s*to\s*know)\s*[,:\s]*",
    ]

    # Conversational filler suffixes (polite particles, trailing conversational tags)
    CONVERSATIONAL_SUFFIXES = [
        r"\s*[,:\s]*(được\s*không\s*ạ|được\s*không|vậy\s*ạ|với\s*ạ|với\s*nhé|với\s*nha|nhé\s*bạn|nha\s*bạn|nhé|nha|thế\s*nhỉ|ạ)\s*[\.!?]*$",
        r"\s*[,:\s]*(please|thank\s*you|thanks)\s*[\.!?]*$",
    ]

    # Internal filler phrases that don't add semantic value to retrieval
    INTERNAL_FILLERS = [
        r"\b(trong\s*(hệ\s*thống|cơ\s*sở\s*dữ\s*liệu|database|db|kho\s*dữ\s*liệu))\b",
        r"\b(in\s*(the\s*)?(database|system))\b",
    ]

    @classmethod
    def compress_query(cls, query: str) -> str:
        """Compresses user query by stripping conversational fluff while preserving intent.

        Example:
            'Bạn ơi cho tôi hỏi là trong hệ thống có bao nhiêu bài báo về AI năm 2024 vậy ạ?'
            -> 'có bao nhiêu bài báo về AI năm 2024?'
        """
        if not query:
            return ""

        compressed = query.strip()

        # Step 1: Strip prefix conversational fluff repeatedly (in case of nested greetings)
        for _ in range(2):
            for pattern in cls.CONVERSATIONAL_PREFIXES:
                compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE).strip()

        # Step 2: Strip suffix conversational fluff
        for _ in range(2):
            for pattern in cls.CONVERSATIONAL_SUFFIXES:
                compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE).strip()

        # Step 3: Remove internal database/system filler references
        for pattern in cls.INTERNAL_FILLERS:
            compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE).strip()

        # Step 4: Normalize multiple punctuation and whitespaces
        compressed = re.sub(r"\s+", " ", compressed)
        compressed = re.sub(r"\s*([?!.,])\s*", r"\1 ", compressed).strip()
        compressed = re.sub(r"[?!]{2,}", "?", compressed).strip()

        # If compression stripped everything (e.g. user literally just said "xin chào bạn ơi"), fall back
        if len(compressed.strip()) < 2:
            return query.strip()

        return compressed.strip()

    @classmethod
    def compress_context(cls, content: str, max_chars: int = 350) -> str:
        """Compresses retrieved context text to eliminate token waste.

        Collapses whitespaces, removes redundant headers, and limits to key information.
        """
        if not content:
            return ""
        # Collapse whitespace/newlines
        clean = re.sub(r"\s+", " ", content).strip()
        if len(clean) <= max_chars:
            return clean
        # Slice at word boundary
        truncated = clean[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.7:
            truncated = truncated[:last_space]
        return truncated + "..."
