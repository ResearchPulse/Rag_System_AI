import unittest
from unittest.mock import patch, MagicMock
from app.modules.retrieval.text_to_sql.engine import TextToSQLEngine
from app.modules.retrieval.text_to_sql.schemas import SQLExecutionResult


class TestTextToSQLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TextToSQLEngine()

    def test_security_guardrail_blocks_mutation(self):
        """Ensures non-SELECT queries are strictly rejected by security validation."""
        destructive_queries = [
            "DROP TABLE \"Article\";",
            "DELETE FROM \"Author\" WHERE author_id = 1;",
            "INSERT INTO \"Article\" (title) VALUES ('Hacked');",
            "UPDATE \"Journal\" SET display_name = 'test';",
            "TRUNCATE \"Keyword\";",
            "ALTER TABLE \"Topic\" ADD COLUMN hacked text;",
        ]
        for sql in destructive_queries:
            val = self.engine.validate_and_sanitize(sql)
            self.assertFalse(val.is_valid, f"Expected {sql} to be rejected")
            self.assertTrue("Only SELECT" in val.error_message or "destructive" in val.error_message)

    def test_security_guardrail_injects_limit(self):
        """Ensures SELECT queries without LIMIT are sanitized with LIMIT 10."""
        sql = 'SELECT title FROM "Article" WHERE publication_year = 2023;'
        val = self.engine.validate_and_sanitize(sql)
        self.assertTrue(val.is_valid)
        self.assertIn("LIMIT", val.sanitized_sql)

    def test_project_scope_injection(self):
        """Ensures project scoping is enforced if missing when project_id is passed."""
        sql = 'SELECT title FROM "Article" WHERE publication_year = 2023 LIMIT 5;'
        val = self.engine.validate_and_sanitize(sql, project_id=18)
        self.assertTrue(val.is_valid)
        self.assertIn("Project_Article_Scope", val.sanitized_sql)
        self.assertIn("18", val.sanitized_sql)

    def test_fallback_sql_generation_entities(self):
        """Ensures fallback SQL generator covers all core entities with and without project scope."""
        queries_and_entities = [
            ("Tác giả nào có trích dẫn cao nhất?", "Author"),
            ("Bài báo nào được trích dẫn nhiều nhất?", "Article"),
            ("Tạp chí nào có nhiều bài báo nhất?", "Journal"),
            ("Chủ đề nào phổ biến nhất?", "Topic"),
            ("Từ khóa nào xuất hiện nhiều nhất?", "Keyword"),
            ("Quốc gia nào có nhiều bài báo nhất?", "Zone"),
        ]
        for q, expected_table in queries_and_entities:
            sql = self.engine._generate_fallback_sql(q, project_id=18)
            self.assertIn(f'"{expected_table}"', sql)
            self.assertIn("18", sql)

    def test_markdown_scalar_count_formatting(self):
        """Ensures scalar COUNT results are formatted clearly."""
        exec_res = SQLExecutionResult(
            sql='SELECT COUNT(*) FROM "Article";',
            columns=["count"],
            rows=[(2137,)],
            row_count=1,
            latency_ms=1.5,
            success=True,
        )
        md = self.engine._build_markdown_output("[Thống kê Đề tài #18]", "Có bao nhiêu bài báo?", exec_res)
        self.assertIn("2,137", md)
        self.assertIn("Count", md)

    def test_markdown_ranking_formatting(self):
        """Ensures tabular ranking results are formatted with top 1 highlight."""
        exec_res = SQLExecutionResult(
            sql='SELECT au.display_name, total_citations, paper_count FROM "Author" au;',
            columns=["author_name", "total_citations", "paper_count"],
            rows=[
                ("Xue Qin Yu", 4520, 15),
                ("John Doe", 3200, 10),
            ],
            row_count=2,
            latency_ms=2.0,
            success=True,
        )
        md = self.engine._build_markdown_output("[Thống kê]", "Tác giả nào có trích dẫn cao nhất?", exec_res)
        self.assertIn("Dẫn đầu (Hạng 1)", md)
        self.assertIn("Xue Qin Yu", md)
        self.assertIn("4,520", md)
        self.assertIn("Hạng 2", md)
        self.assertIn("John Doe", md)

    @patch.object(TextToSQLEngine, "generate_sql")
    @patch.object(TextToSQLEngine, "_execute_query")
    def test_execute_and_format_end_to_end(self, mock_exec, mock_gen):
        """Tests complete execute_and_format pipeline."""
        mock_gen.return_value = 'SELECT display_name, citations FROM "Author" WHERE pas.project_id = 18;'
        mock_exec.return_value = SQLExecutionResult(
            sql='SELECT display_name, citations FROM "Author";',
            columns=["display_name", "citations"],
            rows=[("Dr. Alice", 500)],
            row_count=1,
            latency_ms=3.0,
            success=True,
        )
        chunk = self.engine.execute_and_format("Tác giả trích dẫn cao nhất", project_id=18)
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.metadata.get("source"), "postgresql_sql")
        self.assertEqual(chunk.metadata.get("type"), "Aggregation")
        self.assertEqual(chunk.metadata.get("project_id"), 18)
        self.assertIn("Dr. Alice", chunk.content)


if __name__ == "__main__":
    unittest.main()
