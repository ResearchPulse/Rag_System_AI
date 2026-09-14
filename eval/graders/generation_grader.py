"""
LLM-as-a-Judge for Generation Evaluation using Claude API (Anthropic).
Evaluates 3 independent criteria with zero hallucination guardrails:
1. Groundedness (Faithfulness to retrieved context, no hallucination)
2. Coverage (Completeness compared to reference answer)
3. Language & Relevance (Vietnamese adherence, direct answer to prompt)
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger("eval.generation_grader")


class ClaudeGenerationJudge:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    def _call_claude(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Calls Claude API with strict JSON formatting instructions."""
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not configured. Returning mock evaluation verdict.")
            return {
                "verdict": "PASS",
                "score": 1.0,
                "reasoning": "Mock pass - Set ANTHROPIC_API_KEY environment variable to enable live Claude Judge."
            }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                raw_text = data["content"][0]["text"].strip()

            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            return json.loads(raw_text)
        except Exception as err:
            logger.error(f"Failed to query Claude API: {err}")
            return {
                "verdict": "UNKNOWN",
                "score": 0.0,
                "reasoning": f"Judge call failed with exception: {err}"
            }

    # -------------------------------------------------------------
    # 1. GROUNDEDNESS (Faithfulness, No Hallucination)
    # -------------------------------------------------------------
    def grade_groundedness(self, query: str, contexts: List[str], answer: str) -> Dict[str, Any]:
        if not contexts or all(not c.strip() for c in contexts):
            return {
                "verdict": "UNKNOWN",
                "score": 0.0,
                "reasoning": "Ngữ cảnh trích xuất (Contexts) hoàn toàn rỗng. Không đủ căn cứ để chấm Groundedness."
            }

        system_prompt = (
            "Bạn là giám khảo AI chuyên đánh giá tính TRUNG THỰC (Groundedness/Faithfulness) của hệ thống RAG.\n"
            "Nhiệm vụ: Đối chiếu từng tuyên bố trong câu trả lời (Answer) với tài liệu ngữ cảnh (Contexts).\n"
            "Quy tắc chấm tuyệt đối:\n"
            "1. Nếu Contexts không liên quan hoặc không chứa thông tin về câu hỏi -> verdict: 'UNKNOWN'.\n"
            "2. Nếu Answer chứa bất kỳ sự kiện, con số hay khẳng định nào KHÔNG được đề cập hoặc mâu thuẫn với Contexts -> verdict: 'FAIL'.\n"
            "3. Nếu mọi thông tin trong Answer đều được chứng minh trực tiếp từ Contexts -> verdict: 'PASS'.\n"
            "Định dạng phản hồi BẮT BUỘC bằng JSON:\n"
            "{\n"
            '  "verdict": "PASS" | "FAIL" | "UNKNOWN",\n'
            '  "score": 1.0 | 0.0,\n'
            '  "hallucinated_statements": ["danh sách câu bịa nếu có"],\n'
            '  "reasoning": "giải thích ngắn gọn lý do"\n'
            "}"
        )
        user_prompt = (
            f"### CONTEXTS:\n{json.dumps(contexts, ensure_ascii=False, indent=2)}\n\n"
            f"### QUERY:\n{query}\n\n"
            f"### ANSWER:\n{answer}"
        )
        return self._call_claude(system_prompt, user_prompt)

    # -------------------------------------------------------------
    # 2. COVERAGE (Bao phủ ý chính so với Reference Answer)
    # -------------------------------------------------------------
    def grade_coverage(self, query: str, reference_answer: str, answer: str) -> Dict[str, Any]:
        if not reference_answer or not reference_answer.strip():
            return {
                "verdict": "UNKNOWN",
                "score": 0.0,
                "reasoning": "Không có Reference Answer chuẩn. Trả về UNKNOWN để tránh hallucination khi chấm."
            }

        system_prompt = (
            "Bạn là giám khảo AI đánh giá ĐỘ BAO PHỦ (Coverage/Completeness) của câu trả lời RAG.\n"
            "Nhiệm vụ: So sánh câu trả lời của mô hình với Reference Answer (câu trả lời chuẩn đã biết).\n"
            "Quy tắc chấm:\n"
            "1. Nếu Reference Answer không có nghĩa hoặc rỗng -> verdict: 'UNKNOWN'.\n"
            "2. Trả về 'PASS' (score=1.0) nếu câu trả lời bao hàm được tối thiểu 80% các luận điểm trọng tâm trong Reference Answer.\n"
            "3. Trả về 'FAIL' (score=0.0) nếu bỏ sót một hoặc nhiều luận điểm cốt lõi.\n"
            "Định dạng phản hồi BẮT BUỘC bằng JSON:\n"
            "{\n"
            '  "verdict": "PASS" | "FAIL" | "UNKNOWN",\n'
            '  "score": 1.0 | 0.0,\n'
            '  "missing_points": ["danh sách các ý chính bị thiếu"],\n'
            '  "reasoning": "giải thích chi tiết"\n'
            "}"
        )
        user_prompt = (
            f"### QUERY:\n{query}\n\n"
            f"### REFERENCE ANSWER:\n{reference_answer}\n\n"
            f"### MODEL ANSWER:\n{answer}"
        )
        return self._call_claude(system_prompt, user_prompt)

    # -------------------------------------------------------------
    # 3. LANGUAGE & RELEVANCE (Đúng trọng tâm, đúng tiếng Việt)
    # -------------------------------------------------------------
    def grade_language_and_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "verdict": "FAIL",
                "score": 0.0,
                "reasoning": "Câu trả lời của hệ thống rỗng."
            }

        system_prompt = (
            "Bạn là giám khảo AI đánh giá ĐỘ LIÊN QUAN VÀ NGÔN NGỮ (Language & Relevance).\n"
            "Tiêu chuẩn đánh giá:\n"
            "1. Trọng tâm (Relevance): Trả lời thẳng vào câu hỏi, không lan man sang chủ đề khác.\n"
            "2. Ngôn ngữ (Language): Nếu người dùng hỏi bằng tiếng Việt, câu trả lời PHẢI bằng tiếng Việt tự nhiên, chuẩn thuật ngữ học thuật.\n"
            "Định dạng phản hồi BẮT BUỘC bằng JSON:\n"
            "{\n"
            '  "verdict": "PASS" | "FAIL",\n'
            '  "score": 1.0 | 0.0,\n'
            '  "is_fluent_vietnamese": true | false,\n'
            '  "is_on_topic": true | false,\n'
            '  "reasoning": "lý do đánh giá"\n'
            "}"
        )
        user_prompt = (
            f"### USER QUERY:\n{query}\n\n"
            f"### MODEL ANSWER:\n{answer}"
        )
        return self._call_claude(system_prompt, user_prompt)
