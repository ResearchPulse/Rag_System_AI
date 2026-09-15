"""Stage 4: Embedding Classifier (kNN vs Few-Shot Examples) for Semantic Routing.

Implements lightweight, high-performance in-memory semantic vector classification
using character/word n-gram vector embeddings and cosine similarity kNN.
Zero external network calls, zero API quota cost, robust against typos and colloquial phrasing.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FewShotExample:
    def __init__(self, query: str, category: str, sub_category: str, target_entity: Optional[str] = None):
        self.query = query
        self.category = category
        self.sub_category = sub_category
        self.target_entity = target_entity


# Curated academic prototype examples across all standard routing intents
FEW_SHOT_CORPUS: List[FewShotExample] = [
    # 1. SQL Aggregation & Text-to-SQL (Counts, Metrics, Superlatives, Rankings)
    FewShotExample("Có bao nhiêu bài báo xuất bản năm 2023?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Tổng số lượng bài báo trong cơ sở dữ liệu là bao nhiêu?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Thống kê số lượng bài báo theo từng năm", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Đếm tổng số công bố khoa học từ năm 2020 đến nay", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Tác giả nào có lượng trích dẫn cao nhất?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Ai là tác giả có nhiều trích dẫn nhất trong đề tài?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Top tác giả được trích dẫn nhiều nhất", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tác giả nào có h-index cao nhất?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tác giả nào có số lượng bài báo cao nhất trong project này?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tác giả nào xuất bản nhiều bài báo nhất trong đề tài?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Ai là tác giả có nhiều bài báo nhất?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tổng số tác giả trong project này là bao nhiêu", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Có bao nhiêu tác giả trong đề tài này?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tổng số tác giả trong dự án này là bao nhiêu?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("đang có bao nhiêu author trong project", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("có bao nhiêu author trong project này", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("tổng số author trong dự án", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("bao nhiêu paper trong project", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("đang có bao nhiêu bài báo trong project", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("tác giả nào có nhiều paper nhất", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("author nào có nhiều bài báo nhất", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("bài báo nào có nhiều trích dẫn nhất trong đề tài", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Bài báo nào có lượt trích dẫn cao nhất?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Bài báo nào được trích dẫn nhiều nhất?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Top bài báo có ảnh hưởng và lượt trích dẫn khủng", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Tạp chí nào có nhiều bài báo nhất?", "direct_lookup", "sql_aggregation", "journal"),
    FewShotExample("Tạp chí nào xuất bản nhiều bài báo nhất trong đề tài?", "direct_lookup", "sql_aggregation", "journal"),
    FewShotExample("Top các tạp chí công bố nhiều bài nghiên cứu nhất", "direct_lookup", "sql_aggregation", "journal"),
    FewShotExample("Chủ đề nào có nhiều bài báo nhất?", "direct_lookup", "sql_aggregation", "topic"),
    FewShotExample("Lĩnh vực nào phổ biến nhất trong hệ thống?", "direct_lookup", "sql_aggregation", "topic"),
    FewShotExample("Top chủ đề nghiên cứu được công bố nhiều nhất", "direct_lookup", "sql_aggregation", "topic"),
    FewShotExample("Từ khóa nào xuất hiện nhiều nhất?", "direct_lookup", "sql_aggregation", "keyword"),
    FewShotExample("Top từ khóa học thuật phổ biến nhất", "direct_lookup", "sql_aggregation", "keyword"),
    FewShotExample("Quốc gia nào có nhiều bài báo nhất?", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Nước nào có số lượng nghiên cứu dẫn đầu?", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Thống kê phân bố bài báo theo quốc gia", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Mỹ có bao nhiêu bài báo?", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Mỹ đang có bao nhiêu bài báo?", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Việt Nam có bao nhiêu bài báo trong đề tài?", "direct_lookup", "sql_aggregation", "country"),
    FewShotExample("Có bao nhiêu tác giả trong hệ thống?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Tổng số tác giả nhà khoa học nghiên cứu viên", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Hệ thống hiện tại có bao nhiêu tác giả đang ghi nhận?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Thống kê tổng số lượng tạp chí venue trong hệ thống", "direct_lookup", "sql_aggregation", "journal"),
    FewShotExample("Đếm số lượng chủ đề chuyên ngành nghiên cứu", "direct_lookup", "sql_aggregation", "topic"),
    FewShotExample("How many papers were published in 2024?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Total number of authors indexed in the system", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("How many authors are in this project?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("How many papers are in this project?", "direct_lookup", "sql_aggregation", "article"),
    FewShotExample("Which author has the highest citation count?", "direct_lookup", "sql_aggregation", "author"),
    FewShotExample("Which paper is the most cited article?", "direct_lookup", "sql_aggregation", "article"),

    # 2. Semantic Similarity Search (Vector Search on pgvector)
    FewShotExample("Deep learning backpropagation in neural networks", "direct_lookup", "semantic_similarity"),
    FewShotExample("Retrieval augmented generation architectures and LLM evaluation", "direct_lookup", "semantic_similarity"),
    FewShotExample("Graph neural networks for drug discovery and molecular analysis", "direct_lookup", "semantic_similarity"),
    FewShotExample("Nghiên cứu về mô hình Transformer trong xử lý ảnh y tế", "direct_lookup", "semantic_similarity"),
    FewShotExample("Thuật toán phân loại văn bản tiếng Việt dùng mô hình ngôn ngữ lớn", "direct_lookup", "semantic_similarity"),
    FewShotExample("Các giải pháp bảo mật blockchain và an toàn thông tin mạng", "direct_lookup", "semantic_similarity"),
    FewShotExample("Tối ưu hóa hiệu năng cơ sở dữ liệu vector cho tìm kiếm ngữ nghĩa", "direct_lookup", "semantic_similarity"),

    # 3. Metadata Lookup (DOI, Title exact)
    FewShotExample("Tra cứu thông tin bài báo có mã DOI 10.1016/j.procs.2023.01.001", "direct_lookup", "metadata_lookup"),
    FewShotExample("Tìm bài báo qua mã DOI 10.1109/TPAMI.2022.3180000", "direct_lookup", "metadata_lookup"),
    FewShotExample("Mã định danh DOI bài báo khoa học", "direct_lookup", "metadata_lookup"),

    # 4. Relational Reasoning: Co-authorship
    FewShotExample("Tác giả Xue Qin Yu đã hợp tác với những ai?", "relational_reasoning", "co_authorship", "author"),
    FewShotExample("Ai là đồng tác giả thường xuyên của GS Nguyễn Văn A?", "relational_reasoning", "co_authorship", "author"),
    FewShotExample("Mạng lưới hợp tác đồng tác giả của tác giả này gồm những ai?", "relational_reasoning", "co_authorship", "author"),
    FewShotExample("Who are the frequent co-authors of researcher Andrew Ng?", "relational_reasoning", "co_authorship", "author"),
    FewShotExample("Tìm các nhà khoa học cùng viết bài báo với tác giả Trần Đức", "relational_reasoning", "co_authorship", "author"),

    # 5. Relational Reasoning: Author Publications
    FewShotExample("Tác giả Geoffrey Hinton đã công bố những bài báo nào?", "relational_reasoning", "author_publications", "author"),
    FewShotExample("Danh sách các công trình nghiên cứu của tác giả Lê Văn B", "relational_reasoning", "author_publications", "author"),
    FewShotExample("Các bài báo khoa học được viết bởi tác giả này", "relational_reasoning", "author_publications", "author"),
    FewShotExample("List all published papers written by author Yann LeCun", "relational_reasoning", "author_publications", "author"),

    # 6. Relational Reasoning: Citation Network
    FewShotExample("Những bài báo nào trích dẫn công trình Attention Is All You Need?", "relational_reasoning", "citation_network", "article"),
    FewShotExample("Mạng lưới trích dẫn và các công bố ảnh hưởng lớn nhất", "relational_reasoning", "citation_network", "article"),
    FewShotExample("Bài viết này được trích dẫn bởi những tác giả nào?", "relational_reasoning", "citation_network", "article"),

    # 7. Hybrid: Filtered Graph Traversal
    FewShotExample("Trong các bài báo AI sau năm 2023 tác giả nào hợp tác nhiều nhất?", "hybrid", "filtered_graph"),
    FewShotExample("Những tác giả nào thuộc chủ đề RAG có số lượng đồng tác giả cao nhất năm 2024?", "hybrid", "filtered_graph"),
    FewShotExample("Mạng lưới hợp tác nghiên cứu Machine Learning giai đoạn 2020-2024", "hybrid", "filtered_graph"),

    # 8. Chitchat & Greetings
    FewShotExample("Chào bạn, bạn có thể giúp gì cho tôi?", "chitchat", "chitchat"),
    FewShotExample("Xin chào trợ lý AI, bạn là ai?", "chitchat", "chitchat"),
    FewShotExample("Hello how are you doing today?", "chitchat", "chitchat"),
    FewShotExample("Cảm ơn bạn rất nhiều nhé!", "chitchat", "chitchat"),
    FewShotExample("Tạm biệt hẹn gặp lại", "chitchat", "chitchat"),
]


class EmbeddingClassifier:
    """kNN Embedding Semantic Classifier using char/word n-gram vectorization."""

    def __init__(self, n_neighbors: int = 3, min_similarity_threshold: float = 0.58):
        self.k = n_neighbors
        self.threshold = min_similarity_threshold
        self.vocab: Dict[str, int] = {}
        self.corpus_matrix: Optional[np.ndarray] = None
        self.examples = FEW_SHOT_CORPUS
        self._build_index()

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extracts word unigrams, bigrams, and character 3-grams for typo resilience."""
        text = unicodedata.normalize("NFC", text.lower().strip())
        words = re.findall(r"\w+", text)
        ngrams: List[str] = list(words)
        # Word bigrams
        for i in range(len(words) - 1):
            ngrams.append(f"{words[i]}_{words[i+1]}")
        # Char trigrams
        padded = f"_{text}_"
        for i in range(len(padded) - 2):
            ngrams.append(padded[i:i+3])
        return ngrams

    def _build_index(self):
        """Constructs vector vocabulary and prototype document matrix."""
        # 1. Build vocabulary
        all_doc_ngrams = [self._extract_ngrams(ex.query) for ex in self.examples]
        vocab_set = set()
        for doc in all_doc_ngrams:
            vocab_set.update(doc)
        self.vocab = {gram: idx for idx, gram in enumerate(sorted(vocab_set))}
        dim = len(self.vocab)

        # 2. Build normalized TF vectors
        matrix = np.zeros((len(self.examples), dim), dtype=np.float32)
        for row_idx, doc in enumerate(all_doc_ngrams):
            for gram in doc:
                col_idx = self.vocab.get(gram)
                if col_idx is not None:
                    matrix[row_idx, col_idx] += 1.0
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm

        self.corpus_matrix = matrix

    def _vectorize(self, text: str) -> np.ndarray:
        """Converts query into normalized feature vector."""
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for gram in self._extract_ngrams(text):
            idx = self.vocab.get(gram)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def predict(self, query: str) -> Optional[Tuple[str, str, float, str, Optional[str]]]:
        """Predicts (category, sub_category, confidence, reasoning, target_entity) via kNN.

        Returns None if max similarity is below threshold (cascades to LLM Fallback).
        """
        if not query or self.corpus_matrix is None:
            return None

        q_vec = self._vectorize(query)
        if np.linalg.norm(q_vec) == 0:
            return None

        # Cosine similarity against all few-shot prototypes
        sims = np.dot(self.corpus_matrix, q_vec)
        top_indices = np.argsort(sims)[::-1][: self.k]
        top_sim = float(sims[top_indices[0]])

        if top_sim < self.threshold:
            return None  # Low confidence, let LLM fallback handle it

        best_ex = self.examples[top_indices[0]]
        reasoning = (
            f"Phân loại ngữ nghĩa chính xác qua Stage 4 Embedding Classifier (kNN similarity={top_sim:.2f} "
            f"khớp với mẫu '{best_ex.query}')."
        )
        return best_ex.category, best_ex.sub_category, round(top_sim, 2), reasoning, best_ex.target_entity
