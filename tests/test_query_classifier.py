import unittest
from app.modules.retrieval.query_classifier import (
    QueryClassifier,
    QueryCategory,
    QuerySubCategory,
)
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.schemas import RetrievalRequest


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
        self.assertEqual(res.extracted_filters.year, 2023)

        # Execution check
        resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
        self.assertGreater(len(resp.results), 0)
        self.assertIn("2,137", resp.results[0].content)

    def test_scenario_2_semantic_similarity(self):
        query = "Deep learning and backpropagation in neural networks"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.DIRECT_LOOKUP)
        self.assertEqual(res.sub_category, QuerySubCategory.SEMANTIC_SIMILARITY)
        self.assertTrue(res.execution_plan.requires_vector_search)

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
        self.assertIn("Xue Qin Yu", res.extracted_filters.author)

        # Execution check on Neo4j
        resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
        self.assertGreater(len(resp.results), 0)
        self.assertEqual(resp.results[0].metadata.get("source"), "neo4j")

    def test_scenario_6_hybrid_filtered_graph(self):
        query = "Trong các bài báo xuất bản sau 2023 thuộc chủ đề AI, tác giả nào hợp tác với nhau nhiều nhất?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.HYBRID)
        self.assertEqual(res.sub_category, QuerySubCategory.FILTERED_GRAPH)
        self.assertTrue(res.execution_plan.requires_graph_traversal)
        self.assertTrue(res.execution_plan.requires_vector_search)

    def test_scenario_7_chitchat(self):
        query = "Chào bạn, bạn có thể giúp gì cho tôi?"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.CHITCHAT)
        self.assertEqual(res.sub_category, QuerySubCategory.CHITCHAT)
        self.assertFalse(res.execution_plan.requires_vector_search)
        self.assertFalse(res.execution_plan.requires_graph_traversal)

    def test_scenario_8_clarification_needed(self):
        query = "bài báo"
        res = self.classifier.classify(query)
        self.assertEqual(res.category, QueryCategory.CLARIFICATION_NEEDED)
        self.assertEqual(res.sub_category, QuerySubCategory.AMBIGUOUS)

        resp = self.service.retrieve(RetrievalRequest(query=query, top_k=2))
        self.assertGreater(len(resp.results), 0)
        self.assertEqual(resp.results[0].metadata.get("source"), "system_guardrail")


if __name__ == "__main__":
    unittest.main()
