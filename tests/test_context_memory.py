import unittest
from starlette.testclient import TestClient
from app.main import app
from app.modules.generation.context_memory.service import ContextMemoryService
from app.modules.generation.context_memory.schemas import UpdateContextMemoryRequest


class TestContextMemory(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.service = ContextMemoryService(max_turns=3)
        self.test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        self.test_project_id = 12
        ContextMemoryService._sessions.clear()


    def test_record_turn_and_memory_state(self):
        # 1. Record turn 1
        q1 = "Tìm các bài báo về Graph RAG năm 2025"
        a1 = "Có 3 bài báo nổi bật: **Graph-based Neural Retrieval** và **Hybrid Graph RAG**."
        mem = self.service.record_turn(
            project_id=self.test_project_id,
            user_id=self.test_user_id,
            user_query=q1,
            assistant_answer=a1,
        )
        self.assertEqual(mem.turn_count, 1)
        self.assertEqual(mem.active_year, 2025)
        self.assertIn("Graph-based Neural Retrieval", mem.referenced_articles)

        # 2. Test coreference query reformulation
        followup_q = "Ai là tác giả của bài đầu tiên?"
        reformulated = self.service.reformulate_query_with_context(
            query=followup_q,
            project_id=self.test_project_id,
            user_id=self.test_user_id,
        )
        self.assertIn("Graph-based Neural Retrieval", reformulated)

    def test_format_history_for_prompt(self):
        self.service.record_turn(
            project_id=self.test_project_id,
            user_id=self.test_user_id,
            user_query="Chủ đề AI năm 2025",
            assistant_answer="Xu hướng AI đang phát triển mạnh mẽ.",
        )
        history_str = self.service.format_history_for_prompt(self.test_project_id, self.test_user_id)
        self.assertIn("LỊCH SỬ TRAO ĐỔI TRƯỚC ĐÓ", history_str)
        self.assertIn("Chủ đề AI năm 2025", history_str)

    def test_api_context_endpoints(self):
        # 1. Update context memory via API
        update_payload = {
            "project_id": self.test_project_id,
            "user_id": self.test_user_id,
            "active_topic": "Quantum Computing",
            "active_year": 2026,
            "referenced_articles": ["Quantum Quantum RAG 2026"],
        }
        res_update = self.client.post("/api/v1/chat/context/update", json=update_payload)
        self.assertEqual(res_update.status_code, 200)
        data = res_update.json()
        self.assertEqual(data["active_topic"], "Quantum Computing")
        self.assertEqual(data["active_year"], 2026)

        # 2. Get context memory via API
        res_get = self.client.get(f"/api/v1/chat/context?project_id={self.test_project_id}&user_id={self.test_user_id}")
        self.assertEqual(res_get.status_code, 200)
        get_data = res_get.json()
        self.assertEqual(get_data["active_topic"], "Quantum Computing")

        # 3. Reset context memory via API
        res_reset = self.client.post(f"/api/v1/chat/context/reset?project_id={self.test_project_id}&user_id={self.test_user_id}")
        self.assertEqual(res_reset.status_code, 200)
        self.assertTrue(res_reset.json()["success"])

        # Verify memory is empty after reset
        res_get_after = self.client.get(f"/api/v1/chat/context?project_id={self.test_project_id}&user_id={self.test_user_id}")
        self.assertEqual(res_get_after.json()["turn_count"], 0)


if __name__ == "__main__":
    unittest.main()
