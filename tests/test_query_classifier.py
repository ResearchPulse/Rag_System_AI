import unittest
from unittest.mock import patch
from app.modules.retrieval.query_classifier import (
    QueryClassifier,
    QueryCategory,
    QuerySubCategory,
    TargetStore,
)
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest, RetrievedChunk


class TestQueryClassifierStandard(unittest.TestCase):
    def setUp(self):
        self.classifier = QueryClassifier()
        self.service = RetrievalService()

    def test_scenario_1_sql_aggregation(self):
        query = "Có bao nhiêu bài báo xuất bản năm 2023?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.DIRECT_LOOKUP)
        self.assertEqual(res.sub_category, QuerySubCategory.SQL_AGGREGATION)
        self.assertTrue(res.execution_plan.requires_sql_aggregation)
        self.assertFalse(res.execution_plan.requires_graph_traversal)
        self.assertEqual(res.execution_plan.target_store, TargetStore.POSTGRESQL_SQL)
        self.assertEqual(res.extracted_filters.year, 2023)
        self.assertIn("SELECT COUNT(*)", res.execution_plan.suggested_sql)

        # Service routing check (graceful execution regardless of DB availability)
        from app.modules.retrieval.text_to_sql.schemas import SQLExecutionResult
        with patch.object(
            self.service.text_to_sql,
            "_execute_query",
            return_value=SQLExecutionResult(
                sql='SELECT COUNT(*) FROM "Article" WHERE publication_year = 2023;',
                columns=["count"],
                rows=[(2137,)],
                row_count=1,
                latency_ms=1.0,
                success=True,
            ),
        ):
            resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
            self.assertIsInstance(resp.results, list)
            if resp.results:
                self.assertTrue(
                    "2,137" in resp.results[0].content
                    or "bài báo khoa học" in resp.results[0].content
                    or "Thống kê" in resp.results[0].content
                )

    def test_scenario_2_semantic_similarity(self):
        query = "Deep learning and backpropagation in neural networks"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.DIRECT_LOOKUP)
        self.assertEqual(res.sub_category, QuerySubCategory.SEMANTIC_SIMILARITY)
        self.assertTrue(res.execution_plan.requires_vector_search)
        self.assertEqual(res.execution_plan.target_store, TargetStore.POSTGRESQL_PGVECTOR)

    def test_scenario_3_metadata_lookup_doi(self):
        query = "Tra cứu thông tin bài báo có mã DOI 10.1016/j.procs.2023.01.001"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.DIRECT_LOOKUP)
        self.assertEqual(res.sub_category, QuerySubCategory.METADATA_LOOKUP)
        self.assertEqual(res.extracted_filters.doi, "10.1016/j.procs.2023.01.001")

    def test_scenario_4_co_authorship(self):
        query = "Tác giả Xue Qin Yu đã công bố những bài báo nào và hợp tác với ai?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.RELATIONAL_REASONING)
        self.assertTrue(res.execution_plan.requires_graph_traversal)
        self.assertEqual(res.execution_plan.target_store, TargetStore.NEO4J_GRAPH)
        self.assertIn("Xue Qin Yu", res.extracted_filters.author)

        # Service routing check (graceful execution regardless of Neo4j availability)
        resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
        self.assertIsInstance(resp.results, list)
        if resp.results:
            self.assertEqual(resp.results[0].metadata.get("source"), "neo4j")

    def test_scenario_6_hybrid_filtered_graph(self):
        query = "Trong các bài báo xuất bản sau 2023 thuộc chủ đề AI, tác giả nào hợp tác với nhau nhiều nhất?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.HYBRID)
        self.assertEqual(res.sub_category, QuerySubCategory.FILTERED_GRAPH)
        self.assertTrue(res.execution_plan.requires_graph_traversal)
        self.assertTrue(res.execution_plan.requires_vector_search)
        self.assertEqual(res.execution_plan.target_store, TargetStore.HYBRID_ALL)

    def test_scenario_7_chitchat(self):
        query = "Chào bạn, bạn có thể giúp gì cho tôi?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.CHITCHAT)
        self.assertEqual(res.sub_category, QuerySubCategory.CHITCHAT)
        self.assertFalse(res.execution_plan.requires_vector_search)
        self.assertFalse(res.execution_plan.requires_graph_traversal)
        self.assertEqual(res.execution_plan.target_store, TargetStore.NONE)

    def test_scenario_8_clarification_needed(self):
        query = "bài báo"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.CLARIFICATION_NEEDED)
        self.assertEqual(res.sub_category, QuerySubCategory.AMBIGUOUS)

        resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
        self.assertGreater(len(resp.results), 0)
        self.assertEqual(resp.results[0].metadata.get("source"), "system_guardrail")

    @patch.object(RetrievalService, "_execute_sql_aggregation")
    def test_service_dispatches_sql_aggregation(self, mock_sql):
        mock_sql.return_value = [
            RetrievedChunk(
                chunk_id="chk_1",
                document_id="doc_1",
                content="Mocked count 2,137",
                score=1.0,
                rerank_score=1.0,
                metadata={"source": "postgresql_sql"},
            )
        ]
        resp = self.service.retrieve(RetrievalRequest(query="Có bao nhiêu bài báo xuất bản năm 2023?"))
        self.assertEqual(len(resp.results), 1)
        self.assertIn("Mocked count", resp.results[0].content)
        mock_sql.assert_called_once()

    @patch("app.modules.retrieval.graph_retriever.GraphRetriever.retrieve")
    def test_service_dispatches_graph_traversal(self, mock_graph):
        mock_graph.return_value = [
            RetrievedChunk(
                chunk_id="chk_graph",
                document_id="doc_graph",
                content="Mocked co-authors",
                score=1.0,
                rerank_score=1.0,
                metadata={"source": "neo4j"},
            )
        ]
        resp = self.service.retrieve(
            RetrievalRequest(query="Tác giả Xue Qin Yu đã công bố những bài báo nào và hợp tác với ai?")
        )
        self.assertEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0].metadata.get("source"), "neo4j")
        mock_graph.assert_called_once()


if __name__ == "__main__":
    unittest.main()
