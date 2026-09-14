import unittest
from starlette.testclient import TestClient
from app.main import app
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.project_scope import ProjectScopeService, ProjectScopeMetadata


class TestProjectScopeRAG(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.retrieval_service = RetrievalService()
        self.scope_service = ProjectScopeService()
        self.test_project_id = 12

    def test_project_scope_metadata(self):
        meta = self.scope_service.get_project_metadata(self.test_project_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.project_id, self.test_project_id)
        summary = meta.to_context_summary()
        self.assertIn("Phạm vi Đề tài Nghiên cứu", summary)
        self.assertIn(f"#{self.test_project_id}", summary)

    def test_retrieval_with_project_id(self):
        req = RetrievalRequest(
            query="Các bài báo nổi bật về trí tuệ nhân tạo",
            top_k=5,
            project_id=self.test_project_id,
        )
        res = self.retrieval_service.retrieve(req)
        self.assertEqual(res.query, req.query)
        self.assertGreater(len(res.results), 0)

        # First chunk should be the project scope metadata context
        scope_chunks = [c for c in res.results if c.metadata.get("type") == "project_scope"]
        self.assertTrue(len(scope_chunks) > 0)
        self.assertEqual(scope_chunks[0].metadata.get("project_id"), self.test_project_id)

    def test_chat_pipeline_with_project_scope(self):
        chat_req = {
            "query": "Tổng quan xu hướng và các bài báo trong đề tài này",
            "project_id": self.test_project_id,
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "save_history": True,
        }
        res = self.client.post("/api/v1/chat", json=chat_req)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIsNotNone(data.get("user_message_id"))
        self.assertIsNotNone(data.get("assistant_message_id"))


if __name__ == "__main__":
    unittest.main()
