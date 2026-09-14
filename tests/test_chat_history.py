import unittest
from starlette.testclient import TestClient
from app.main import app
from app.api.deps import get_chat_history_service
from app.modules.chat_history.schemas import ChatMessageCreate, ChatMessageRole, ChatMessageStatus


class TestChatHistoryAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.service = get_chat_history_service()
        self.test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        self.test_project_id = 999


    def test_create_and_get_chat_message(self):
        # 1. Create message
        payload = {
            "project_id": self.test_project_id,
            "user_id": self.test_user_id,
            "role": "USER",
            "content": "Xu hướng nghiên cứu Graph RAG năm 2026?",
        }
        res = self.client.post("/api/v1/chat/messages", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["content"], payload["content"])
        self.assertEqual(data["role"], "USER")
        msg_id = data["message_id"]

        # 2. Get message detail
        res_detail = self.client.get(f"/api/v1/chat/messages/{msg_id}?project_id={self.test_project_id}&user_id={self.test_user_id}")
        self.assertEqual(res_detail.status_code, 200)
        detail_data = res_detail.json()
        self.assertEqual(detail_data["message_id"], msg_id)

    def test_get_chat_history_pagination(self):
        # Create a turn
        u_id, a_id = self.service.record_chat_turn(
            project_id=self.test_project_id,
            user_id=self.test_user_id,
            user_query="Câu hỏi test 1",
            assistant_answer="Câu trả lời test 1",
            model="llama3.2:3b",
        )
        self.assertIsNotNone(u_id)
        self.assertIsNotNone(a_id)

        # Retrieve history
        res = self.client.get(
            f"/api/v1/chat/history?project_id={self.test_project_id}&user_id={self.test_user_id}&limit=10&offset=0&order=asc"
        )
        self.assertEqual(res.status_code, 200)
        history_data = res.json()
        self.assertGreaterEqual(history_data["total"], 2)
        self.assertGreaterEqual(len(history_data["messages"]), 2)

    def test_chat_pipeline_auto_record_history(self):
        # Test calling /api/v1/chat with save_history=True and user_id
        chat_req = {
            "query": "Xin chào trợ lý",
            "project_id": self.test_project_id,
            "user_id": self.test_user_id,
            "save_history": True,
        }
        res = self.client.post("/api/v1/chat", json=chat_req)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIsNotNone(data.get("user_message_id"))
        self.assertIsNotNone(data.get("assistant_message_id"))

        # Verify messages exist in history
        u_msg_id = data["user_message_id"]
        res_get = self.client.get(f"/api/v1/chat/messages/{u_msg_id}?user_id={self.test_user_id}")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["content"], "Xin chào trợ lý")

    def test_clear_chat_history(self):
        # Clear history
        res_del = self.client.delete(
            f"/api/v1/chat/history?project_id={self.test_project_id}&user_id={self.test_user_id}"
        )
        self.assertEqual(res_del.status_code, 200)
        del_data = res_del.json()
        self.assertGreaterEqual(del_data["deleted_count"], 0)


if __name__ == "__main__":
    unittest.main()
