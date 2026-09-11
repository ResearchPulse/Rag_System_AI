import unittest
from starlette.testclient import TestClient
from app.main import app


class TestModularMonolithAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["architecture"], "Modular Monolith")

    def test_chat_pipeline(self):
        response = self.client.post("/api/v1/chat", json={"query": "AI research trends 2026"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertEqual(data["query"], "AI research trends 2026")


if __name__ == "__main__":
    unittest.main()
