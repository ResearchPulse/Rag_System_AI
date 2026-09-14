"""
Robustness Grader for Out-of-Domain (OOD) & Unanswerable questions.
Verifies that the RAG pipeline explicitly refuses or declares lack of information
instead of hallucinating an answer.
"""
import re
from typing import Dict, Any, List

DEFAULT_REFUSAL_PATTERNS = [
    r"không tìm thấy",
    r"không có thông tin",
    r"không có dữ liệu",
    r"nằm ngoài phạm vi",
    r"không thể trả lời",
    r"chưa có tài liệu",
    r"không tìm thấy bất kỳ",
    r"không có bài báo nào",
    r"không tìm thấy.*(bài báo|tác giả|thông tin|tài liệu)",
]


class RobustnessGrader:
    def __init__(self, refusal_patterns: List[str] = None):
        self.refusal_patterns = refusal_patterns or DEFAULT_REFUSAL_PATTERNS

    def evaluate(self, answer: str) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "passed": False,
                "verdict": "FAIL",
                "matched_phrases": [],
                "reasoning": "Câu trả lời rỗng."
            }

        answer_lower = answer.lower()
        matched = []
        for pat in self.refusal_patterns:
            if re.search(pat, answer_lower):
                matched.append(pat)

        passed = len(matched) > 0
        return {
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
            "matched_phrases": matched,
            "reasoning": (
                f"Hệ thống đã từ chối hợp lệ khi gặp câu hỏi OOD (khớp mẫu: {matched})."
                if passed
                else "Hệ thống KHÔNG từ chối câu hỏi ngoài miền dữ liệu, có nguy cơ bịa đặt (hallucination)."
            )
        }

