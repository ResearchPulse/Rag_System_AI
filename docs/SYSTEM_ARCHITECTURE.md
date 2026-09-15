# KIẾN TRÚC VÀ TÀI LIỆU TOÀN DIỆN HỆ THỐNG RAG_SYSTEM_AI
**Scientific Journal Publication Trend Tracking RAG System**
*Kiến trúc Đơn khối dạng Mô-đun (Modular Monolith)*

---

## MỤC LỤC
1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Công nghệ & Cơ sở hạ tầng sử dụng](#2-công-nghệ--cơ-sở-hạ-tầng-sử-dụng)
3. [Cấu trúc mã nguồn toàn bộ hệ thống](#3-cấu-trúc-mã-nguồn-toàn-bộ-hệ-thống)
4. [Chu trình RAG Pipeline chuẩn hóa (12 Bước qua 4 Giai đoạn)](#4-chu-trình-rag-pipeline-chuẩn-hóa-12-bước-qua-4-giai-đoạn)
5. [Cơ chế hoạt động chi tiết (Under the Hood)](#5-cơ-chế-hoạt-động-chi-tiết-under-the-hood)
   - [5.1. Phân loại câu hỏi & Định tuyến thông minh (Query Classifier)](#51-phân-loại-câu-hỏi--định-tuyến-thông-minh-query-classifier)
   - [5.2. Giới hạn phạm vi đề tài (Project Scope Filtering)](#52-giới-hạn-phạm-vi-đề-tài-project-scope-filtering)
   - [5.3. Truy xuất kết hợp Hybrid (Lexical + pgvector + Neo4j Cypher)](#53-truy-xuất-kết-hợp-hybrid-lexical--pgvector--neo4j-cypher)
   - [5.4. Quản lý ngữ cảnh hội thoại & Giải quyết từ thay thế (Context Memory)](#54-quản-lý-ngữ-cảnh-hội-thoại--giải-quyết-từ-thay-thế-context-memory)
   - [5.5. Cơ chế chịu lỗi cao (Fail-Safe & Fallback Stores)](#55-cơ-chế-chịu-lỗi-cao-fail-safe--fallback-stores)
6. [Flow hoạt động (Workflow & Sequence Diagrams)](#6-flow-hoạt-động-workflow--sequence-diagrams)
   - [Flow 1: Xử lý End-to-End RAG Chat (`/api/v1/chat`)](#flow-1-xử-lý-end-to-end-rag-chat-apiv1chat)
   - [Flow 2: Tiếp nhận & Phân tách tài liệu (Ingestion Pipeline)](#flow-2-tiếp-nhận--phân-tách-tài-liệu-ingestion-pipeline)
   - [Flow 3: Định tuyến truy xuất (Retrieval Routing Pipeline)](#flow-3-định-tuyến-truy-xuất-retrieval-routing-pipeline)
7. [Danh mục toàn bộ API Endpoints (API Reference)](#7-danh-mục-toàn-bộ-api-endpoints-api-reference)
8. [Hướng dẫn cài đặt & Khởi chạy](#8-hướng-dẫn-cài-đặt--khởi-chạy)
9. [Đánh giá chất lượng RAG (Evaluation & Testing)](#9-đánh-giá-chất-lượng-rag-evaluation--testing)

---

## 1. Tổng quan hệ thống

**Rag_System_AI** là lõi xử lý trí tuệ nhân tạo (RAG - Retrieval-Augmented Generation) phục vụ theo dõi, phân tích và giải đáp thông minh về xu hướng công bố các bài báo khoa học quốc tế và trong nước. 

### Mục tiêu cốt lõi:
- **Truy xuất chính xác (High Precision)**: Kết hợp tìm kiếm ngữ nghĩa sâu (Dense Vector) và tìm kiếm từ khóa học thuật chính xác (Lexical / BM25 / Trigram).
- **Khám phá mối quan hệ phức hợp (Relational Knowledge)**: Khai thác đồ thị tri thức (Knowledge Graph) để phân tích mạng lưới tác giả, đồng tác giả, trích dẫn chéo, và các chủ đề liên ngành.
- **Phân tách theo từng đề tài nghiên cứu (Project Scoping)**: Cho phép cô lập kho dữ liệu tri thức theo từng đề tài nghiên cứu cá nhân/nhóm của nhà khoa học (`project_id`).
- **Giao tiếp liên tục (Conversational Multi-turn RAG)**: Duy trì trí nhớ hội thoại trượt (Context Memory) để hiểu câu hỏi kế tiếp (coreference resolution) mà không mất ngữ cảnh.
- **Kiến trúc Modular Monolith**: Đóng gói trong một tiến trình duy nhất (FastAPI) nhưng ranh giới module phân định rõ rệt. Giúp loại bỏ độ trễ mạng giữa các vi dịch vụ (zero network hop latency), đơn giản hóa triển khai cục bộ và qua Docker.

---

## 2. Công nghệ & Cơ sở hạ tầng sử dụng

| Tầng công nghệ | Công nghệ / Thư viện | Vai trò & Mục đích sử dụng |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** `^0.111.0` | Xây dựng RESTful API bất đồng bộ tốc độ cao, tự động sinh chuẩn tài liệu OpenAPI / Swagger UI. |
| **Web Server** | **Uvicorn** `^0.30.0` | ASGI Web Server hiệu năng cao cho Python. |
| **Dữ liệu & Cấu hình** | **Pydantic v2** & **pydantic-settings** | Xác thực kiểu dữ liệu (Type validation), tuần tự hóa Schema và quản lý biến môi trường từ `.env`. |
| **Vector & Metadata DB** | **PostgreSQL** (với extension **pgvector**) | - Quản lý bảng dữ liệu bài báo `Article`, tác giả, phạm vi đề tài `Project_Article_Scope`.<br>- Lưu trữ `embedding` (vector 768 chiều) và tìm kiếm độ tương đồng cosine (`<=>`).<br>- Lưu trữ lịch sử tin nhắn `Project_Chat_Message`. |
| **Knowledge Graph DB** | **Neo4j** (`neo4j ^5.20.0`) | - Lưu trữ đồ thị tri thức dạng Graph Node (`Article`, `Author`, `Topic`, `Journal`).<br>- Truy vấn quan hệ phức hợp bằng ngôn ngữ Cypher (`WRITES`, `HAS_TOPIC`, `PUBLISHED_IN`, `CITES`). |
| **DB Connectors** | **asyncpg**, **psycopg2-binary** | Kết nối PostgreSQL trực tiếp với cơ chế connection pool, fallback và retry khi có sự cố mạng. |
| **Embedding Engine** | **Google Gemini Embeddings** (`gemini-embedding-001`) / Mock | Chuyển đổi văn bản thành vector nhúng ngữ nghĩa mật độ cao 768 chiều. |
| **LLM Provider** | **Local Ollama** (`llama3.2:3b`) / Cloud Fallback (**Gemini 1.5**, **OpenAI**) | Sinh câu trả lời bằng tiếng Việt, căn cứ chặt chẽ vào tài liệu trích xuất (grounded answer generation), có trích dẫn nguồn. |
| **Đánh giá & Kiểm thử**| **Pytest**, **unittest**, Custom Graders | Bộ test suite tự động kiểm thử Project Scope, API contracts, Retrieval Graders (hit rate, MRR, latency). |
| **Đóng gói & DevOps** | **Poetry**, **Docker**, **Docker Compose** | Quản lý phụ thuộc cô lập và triển khai đồng bộ hạ tầng app + db. |

---

## 3. Cấu trúc mã nguồn toàn bộ hệ thống

```
Rag_System_AI/
├── app/
│   ├── main.py                               # FastAPI App entrypoint, cấu hình CORS, routing
│   ├── api/
│   │   ├── deps.py                           # Dependency Injection (lru_cache) cho toàn bộ services
│   │   └── v1/
│   │       ├── router.py                     # Gom tất cả endpoint routers v1
│   │       └── endpoints/
│   │           ├── ingestion.py              # POST/GET /api/v1/documents
│   │           ├── embedding.py              # POST /api/v1/embeddings
│   │           ├── retrieval.py              # POST /api/v1/retrieve & /retrieve/classify
│   │           ├── generation.py             # POST /api/v1/generate
│   │           ├── rag.py                    # POST /api/v1/chat (Unified End-to-End Pipeline)
│   │           ├── chat_history.py           # Quản lý lịch sử chat (GET/POST/DELETE)
│   │           └── context_memory.py         # Quản lý bộ nhớ hội thoại làm việc (GET/UPDATE/RESET)
│   ├── core/
│   │   ├── config.py                         # Settings qua pydantic-settings đọc từ .env
│   │   └── logger.py                         # Cấu hình logging chuẩn hóa
│   ├── shared/
│   │   ├── schemas.py                        # BaseResponse, ErrorResponse, HealthCheckResponse
│   │   └── pipeline.py                       # Định nghĩa chuẩn tắc 12 bước RAG qua 4 Phase
│   └── modules/                              # 4 Phase RAG cốt lõi + Tiện ích hỗ trợ
│       ├── ingestion/                        # [PHASE 1: INGESTION]
│       │   ├── data_sources/                 # Step 1: Nguồn dữ liệu (PDF, API, Web, Transcripts)
│       │   ├── loaders/                      # Step 2: Đọc và parse định dạng tài liệu
│       │   ├── chunking/                     # Step 3: Phân đoạn ngữ nghĩa (Semantic Chunking)
│       │   ├── metadata/                     # Step 4: Trích xuất metadata (Title, Year, DOI,...)
│       │   ├── schemas.py                    # Ingestion Request/Response models
│       │   └── service.py                    # IngestionService điều phối chunking & trích xuất
│       ├── indexing/                         # [PHASE 2: INDEXING]
│       │   ├── embedders/                    # Step 5: Tạo Vector Embeddings
│       │   ├── vector_store/                 # Step 6: Lưu trữ vector (pgvector)
│       │   ├── graph_store/                  # Lưu trữ đồ thị tri thức (Neo4jGraphStore)
│       │   ├── schemas.py                    # Indexing Request/Response models
│       │   └── service.py                    # IndexingService điều phối
│       ├── retrieval/                        # [PHASE 3: RETRIEVAL]
│       │   ├── query_classifier.py           # Phân loại câu hỏi & phát hiện thực thể
│       │   ├── project_scope.py              # Service quản lý phạm vi nghiên cứu theo đề tài
│       │   ├── graph_retriever.py            # Truy vấn đồ thị tri thức Neo4j bằng Cypher
│       │   ├── query_rewriting/              # Step 7: Viết lại / Nén truy vấn (Query Compressor)
│       │   ├── hybrid_search/                # Step 8: Kết hợp Full-Text / Lexical + Dense Vector
│       │   ├── reranking/                    # Step 9: Tái xếp hạng Cross-Encoder
│       │   ├── schemas.py                    # Retrieval models & Filters
│       │   └── service.py                    # RetrievalService điều phối định tuyến
│       ├── generation/                       # [PHASE 4: GENERATION & EVALUATION]
│       │   ├── context_assembly/             # Step 10: Tập hợp, nén và khử trùng lặp context
│       │   ├── llm/                          # Step 11: Giao tiếp mô hình ngôn ngữ (Ollama/Cloud)
│       │   ├── evaluation/                   # Step 12: Đánh giá Faithfulness, Citation, Latency
│       │   ├── context_memory/               # Quản lý bộ nhớ hội thoại trượt (Working Memory)
│       │   │   ├── schemas.py
│       │   │   └── service.py
│       │   ├── schemas.py                    # Generation Request/Response models
│       │   └── service.py                    # GenerationService thực thi RAG Pipeline
│       └── chat_history/                     # Quản lý lưu trữ lịch sử trò chuyện
│           ├── repository.py                 # Truy vấn bảng Project_Chat_Message (kèm in-memory fallback)
│           ├── schemas.py                    # ChatMessage models
│           └── service.py                    # ChatHistoryService
├── eval/                                     # Khung đánh giá tự động chất lượng RAG
│   ├── run_eval.py                           # Runner thực thi bộ bài test định lượng
│   ├── graders/                              # Bộ chấm điểm Retrieval & Generation
│   ├── tasks/                                # Bộ câu hỏi chuẩn mẫu theo nhiều kịch bản
│   └── report/                               # Báo cáo kết quả đánh giá chi tiết
├── docs/                                     # Tài liệu hệ thống & API Specification
│   ├── SYSTEM_ARCHITECTURE.md                # Tài liệu này
│   ├── TESTING_GUIDE.md                      # Cẩm nang chạy kiểm thử & kiểm chứng RAG
│   └── openapi/openapi.json                  # Contract JSON đặc tả OpenAPI 3.0
├── scripts/
│   └── export_openapi.py                     # Script xuất file openapi.json tự động
├── tests/                                    # Bộ Unit & Integration Tests
│   ├── test_api.py
│   └── test_project_scope.py
├── Dockerfile                                # Single-stage Dockerfile triển khai app
├── docker-compose.yml                        # Docker Compose cấu hình RAG App + DB Network
├── pyproject.toml                            # Quản lý phụ thuộc Poetry
└── .env.example                              # Mẫu biến môi trường
```

---

## 4. Chu trình RAG Pipeline chuẩn hóa (12 Bước qua 4 Giai đoạn)

Hệ thống tuân thủ nghiêm ngặt 12 bước tiêu chuẩn được chia thành 4 giai đoạn lớn nhằm loại bỏ **4 Điểm Lỗi Chí Mạng (Four Failure Points)** thường gặp trong các hệ thống RAG truyền thống:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: INGESTION ("Garbage In, Garbage Out - Xử lý nguồn & Metadata tỉ mỉ")       │
│  ├── [Bước 1: Data Sources]         -> Đa nguồn dữ liệu (PDF, API, Web, Transcripts)   │
│  ├── [Bước 2: Document Loading]     -> Trích xuất văn bản thô, làm sạch ký tự lạ       │
│  ├── [Bước 3: Meaningful Chunking]  -> Phân tách theo cấu trúc ngữ nghĩa (đoạn/mục)    │
│  └── [Bước 4: Metadata Extraction]  -> Trích xuất tên bài báo, năm, tác giả, DOI       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: INDEXING ("Building the Brain - Xây dựng bộ nhớ Vector & Đồ thị")          │
│  ├── [Bước 5: Embeddings]           -> Vector hóa văn bản bằng Gemini Embedding        │
│  └── [Bước 6: Vector & Graph DB]    -> Nạp vào PostgreSQL (pgvector) & Neo4j Graph DB  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: RETRIEVAL ("The Make-or-Break Stage - Truy xuất đúng và trúng")           │
│  ├── [Bước 7: Query Classification] -> Phân loại Intent & Nén truy vấn (Query Rewriting)│
│  ├── [Bước 8: Hybrid Search]        -> Kết hợp Vector Search + Lexical + Graph Cypher  │
│  └── [Bước 9: Reranking]            -> Chấm điểm & tái xếp hạng Context               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: GENERATION & EVALUATION ("Close the Loop - Câu trả lời có căn cứ học thuật")│
│  ├── [Bước 10: Context Assembly]    -> Gom ngữ cảnh, nén context, ghép bộ nhớ hội thoại │
│  ├── [Bước 11: LLM Generation]      -> LLM (Ollama) sinh phản hồi, bắt buộc trích dẫn  │
│  └── [Bước 12: Evaluation]          -> Đo lường Faithfulness, Citation, Token & Độ trễ │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
                                  [CÂU TRẢ LỜI ĐÁNG TIN CẬY]
                   (Có dẫn chứng, trúng đích, phản hồi nhanh, bảo toàn ngữ cảnh)
```

---

## 5. Cơ chế hoạt động chi tiết (Under the Hood)

### 5.1. Phân loại câu hỏi & Định tuyến thông minh (Query Classifier)
Mỗi truy vấn người dùng đi vào hệ thống trước hết sẽ được đưa qua `QueryClassifier` (`app/modules/retrieval/query_classifier.py`):
1. **Phát hiện Chitchat (Chào hỏi, cảm ơn, giới thiệu)**: Trả về lời chào điều hướng mà không cần tốn tài nguyên truy vấn DB.
2. **Phát hiện Câu hỏi thiếu thông tin (`CLARIFICATION_NEEDED`)**: Nếu câu hỏi quá mơ hồ, hệ thống phát cảnh báo và đề nghị người dùng cung cấp thêm thực thể cụ thể.
3. **Phát hiện Tra cứu mã định danh (`METADATA_LOOKUP`)**: Nếu câu hỏi chứa DOI hoặc mã số bài báo, hệ thống ưu tiên thực hiện truy vấn SQL khớp chính xác.
4. **Phát hiện Yêu cầu Thống kê (`SQL_AGGREGATION`)**: Với các câu hỏi đếm số lượng (ví dụ: *"Có bao nhiêu bài báo năm 2024?"*), hệ thống chuyển sang tính toán trực tiếp trên SQL thay vì sinh đoán mò qua Vector.
5. **Phân biệt Truy xuất Ngữ nghĩa vs. Quan hệ đồ thị**:
   - Truy vấn về nội dung nghiên cứu, phương pháp, kết quả -> **Vector Search (pgvector)** kết hợp **Full-Text Search**.
   - Truy vấn về mạng lưới tác giả, đồng tác giả, ai viết nhiều bài báo nhất trong một chủ đề -> **Neo4j Graph Cypher Traversal**.
   - Truy vấn kết hợp (cả nội dung và mạng lưới) -> **Hybrid Search**.

### 5.2. Giới hạn phạm vi đề tài (Project Scope Filtering)
Hệ thống hỗ trợ cơ chế đa người dùng và đa đề tài nghiên cứu:
- Khi người dùng gửi kèm `project_id`, `ProjectScopeService` truy vấn bảng `Project_Article_Scope` để lấy metadata đề tài (Lĩnh vực chuyên môn, Từ khóa trọng tâm, Số lượng bài báo trong phạm vi).
- Toàn bộ câu lệnh tìm kiếm SQL và Vector được tiêm mệnh đề lọc:
  ```sql
  AND article_id IN (SELECT pas.article_id FROM "Project_Article_Scope" pas WHERE pas.project_id = %s)
  ```
- Nhờ đó, câu trả lời của AI được bảo đảm tính biệt lập, không bị rò rỉ dữ liệu ngoài đề tài nghiên cứu của người dùng.

### 5.3. Truy xuất kết hợp Hybrid (Lexical + pgvector + Neo4j Cypher)
Để đạt độ chuẩn xác tối đa:
1. **Lexical Matching**: Tìm kiếm trực tiếp theo danh sách thực thể công nghệ (`distinctive_tokens`), tên bài báo trích dẫn dạng Latinh, và từ khóa chuyên ngành trong cơ sở dữ liệu.
2. **Dense Vector Search**: Sử dụng mô hình nhúng Gemini tạo vector cho câu hỏi người dùng, sau đó tính toán khoảng cách cosine trên PostgreSQL:
   ```sql
   SELECT article_id, title, abstract, publication_year,
          1 - (embedding <=> %s::vector) AS score
   FROM "Article"
   WHERE embedding IS NOT NULL ... AND (1 - (embedding <=> %s::vector)) >= threshold
   ORDER BY embedding <=> %s::vector LIMIT ...
   ```
3. **Neo4j Cypher Traversal**: Khai thác các liên kết đồ thị:
   ```cypher
   MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower($term)
   MATCH (art:Article)-[:HAS_TOPIC]->(t)
   MATCH (a:Author)-[:WRITES]->(art)
   RETURN a.name AS author_name, count(DISTINCT art) AS paper_count
   ORDER BY paper_count DESC LIMIT $limit
   ```

### 5.4. Quản lý ngữ cảnh hội thoại & Giải quyết từ thay thế (Context Memory)
Tại `app/modules/generation/context_memory/service.py`:
- Hệ thống duy trì một bộ nhớ trượt lưu tối đa **5 lượt đối thoại gần nhất** theo cặp `(project_id, user_id)`.
- Khi người dùng hỏi các câu liên tiếp chứa đại từ thay thế (ví dụ: *"Bài báo đó xuất bản năm nào?"*, *"Tác giả của nghiên cứu trên là ai?"*), hàm `reformulate_query_with_context` sẽ tự động ghép thực thể (tên bài báo hoặc tác giả) từ lượt trước vào câu hỏi hiện tại trước khi đưa vào bộ máy tìm kiếm.
- Sau khi sinh phản hồi, các thực thể chính được trích xuất và cập nhật lại vào `ContextMemoryState`.

### 5.5. Cơ chế chịu lỗi cao (Fail-Safe & Fallback Stores)
Hệ thống được thiết kế để không bao giờ bị sập (crash) ngay cả khi hạ tầng phụ thuộc gặp sự cố:
- **PostgreSQL Offline**: `ChatHistoryRepository` và `ProjectScopeService` tự động chuyển sang lưu trữ tạm trong bộ nhớ đệm (In-Memory Fallback) và bật cờ cache timeout 15 giây để tránh chặn luồng I/O.
- **Neo4j Offline**: `GraphRetriever` ghi log cảnh báo và hạ cấp mượt sang Vector / Lexical search.
- **Ollama / LLM Offline**: `GenerationService` tự động chuyển sang chế độ dự phòng tổng hợp trích đoạn trực tiếp từ kết quả tìm kiếm để người dùng vẫn nhận được thông tin tóm tắt kèm dẫn chứng.

---

## 6. Flow hoạt động (Workflow & Sequence Diagrams)

### Flow 1: Xử lý End-to-End RAG Chat (`/api/v1/chat`)

```
Người dùng / Client
      │
      ▼  [POST /api/v1/chat] (query, project_id, user_id, save_history)
┌─────────────────────────────────────────────────────────────┐
│ 1. FastAPI Endpoint (app/api/v1/endpoints/rag.py)            │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Context Memory Resolution                                │
│    - Kiểm tra lượt hội thoại trước                          │
│    - Tự động thay thế đại từ (Coreference Resolution)       │
│    - Rút trích history context ngắn gọn                     │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Phase 3: Retrieval Service                               │
│    - Query Classification (Chitchat? DOI? SQL? Graph? Vec?) │
│    - Lấy thông tin Project Scope (nếu có project_id)        │
│    - Tìm kiếm kết hợp (Lexical + pgvector + Neo4j Cypher)   │
│    - Xếp hạng và cắt tỉa Top-K đoạn trích                   │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Phase 4: Context Assembly & LLM Generation               │
│    - Chuẩn hóa tối đa 4 blocks context, nén ký tự           │
│    - Xây dựng Academic Prompt kèm quy tắc dẫn chứng         │
│    - Gọi LLM (Local Ollama / Cloud API) sinh câu trả lời    │
│    - Thu thập danh mục bài báo đã trích dẫn (citations)     │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Cập nhật trạng thái sau sinh                             │
│    - Cập nhật Context Memory (active_topic, articles)       │
│    - Ghi lịch sử tin nhắn (bảng Project_Chat_Message)       │
│    - Tính toán độ trễ chi tiết (retrieval, gen, total ms)   │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
Client nhận RagPipelineResponse (answer, citations, latency_breakdown, message_ids)
```

---

### Flow 2: Tiếp nhận & Phân tách tài liệu (Ingestion Pipeline)

```
Nguồn tài liệu (PDF, Văn bản, Báo cáo)
      │
      ▼  [POST /api/v1/documents]
┌─────────────────────────────────────────────────────────────┐
│ 1. Document Loader (app/modules/ingestion/loaders/)         │
│    - Làm sạch khoảng trắng, chuẩn hóa mã UTF-8              │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Semantic Chunking (app/modules/ingestion/chunking/)      │
│    - Cắt nhỏ văn bản theo chunk_size (500) & overlap (50)   │
│    - Đảm bảo ranh giới câu nguyên vẹn                       │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Metadata Extraction (app/modules/ingestion/metadata/)    │
│    - Trích xuất tiêu đề, năm công bố, tác giả, DOI          │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
Client nhận DocumentUploadResponse (document_id, total_chunks, chunks metadata)
```

---

### Flow 3: Định tuyến truy xuất (Retrieval Routing Pipeline)

```
Truy vấn từ người dùng
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ QueryClassifier: Phân tích Intent, Category, SubCategory, Thực thể │
└─────┬──────────────┬──────────────┬──────────────┬──────────────┘
      │              │              │              │
      ▼ (Chitchat)   ▼ (DOI match)  ▼ (Agg/Count)  ▼ (Graph/Vec)
 [Chào hỏi xã giao] [SQL tra cứu DOI] [SQL Aggregation] ┌───────────────┐
      │              │              │              │ Lọc theo      │
      │              │              │              │ Project Scope │
      │              │              │              └───────┬───────┘
      │              │              │                      │
      │              │              │            ┌─────────┴─────────┐
      │              │              │            ▼                   ▼
      │              │              │      [Neo4j Cypher]     [pgvector & Lexical]
      │              │              │      (Mạng lưới tác giả,  (Nội dung tóm tắt,
      │              │              │       chủ đề, trích dẫn)   phương pháp, KQ)
      │              │              │            └─────────┬─────────┘
      │              │              │                      │
      └──────────────┴──────────────┴──────────────────────▼
                                            Gộp & Xếp hạng kết quả
                                           (Trả về List[RetrievedChunk])
```

---

## 7. Danh mục toàn bộ API Endpoints (API Reference)

Tất cả các API được công khai tại cổng `http://localhost:8000`. Bạn có thể thử nghiệm trực tiếp qua Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs).

### Nhóm 1: Hệ thống & Kiểm tra sức khỏe (System)
| Method | Endpoint | Tên chức năng | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Application Health Check | Trả về trạng thái hoạt động của service, phiên bản (0.1.0) và môi trường. |
| `GET` | `/` | Root Welcome & Sitemap | Thông tin chào mừng, mô tả kiến trúc và danh mục đường dẫn tài liệu. |

---

### Nhóm 2: Giai đoạn 1 - Nạp & Phân tách tài liệu (Phase 1: Ingestion)
| Method | Endpoint | Request Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/documents` | `DocumentUploadRequest`:<br>- `title` (str)<br>- `content` (str)<br>- `source` (str, optional)<br>- `metadata` (dict, optional)<br>- `chunk_size` (int, default 500)<br>- `chunk_overlap` (int, default 50) | `DocumentUploadResponse`:<br>- `document_id`<br>- `title`<br>- `total_chunks`<br>- `chunks` (list)<br>- `status` ("processed") | Tiếp nhận nội dung văn bản khoa học, thực hiện chia nhỏ ngữ nghĩa và trích xuất siêu dữ liệu. |
| `GET` | `/api/v1/documents/{document_id}` | Path param: `document_id` (str) | `DocumentDetailResponse`:<br>- Toàn bộ thông tin tài liệu và danh sách chi tiết các chunks đã phân tách. | Tra cứu chi tiết một tài liệu đã nạp vào bộ nhớ. |

---

### Nhóm 3: Giai đoạn 2 - Tạo Vector Nhúng (Phase 2: Indexing)
| Method | Endpoint | Request Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/embeddings` | `EmbeddingRequest`:<br>- `texts` (List[str])<br>- `model` (str, optional) | `EmbeddingResponse`:<br>- `model`<br>- `embeddings` (List[List[float]])<br>- `dimension` (768)<br>- `total_tokens` | Chuyển đổi một danh sách các chuỗi văn bản thành các vector số thực mật độ cao. |

---

### Nhóm 4: Giai đoạn 3 - Phân loại & Truy xuất (Phase 3: Retrieval)
| Method | Endpoint | Request Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/retrieve` | `RetrievalRequest`:<br>- `query` (str)<br>- `top_k` (int, default 5)<br>- `project_id` (int, optional)<br>- `filter` (RetrievalFilter, optional) | `RetrievalResponse`:<br>- `query`<br>- `total_found`<br>- `results` (List[RetrievedChunk])<br>- `latency_ms` | Thực hiện định tuyến thông minh: Full-text search, Vector search pgvector, và Neo4j Cypher Traversal theo phạm vi đề tài. |
| `POST` | `/api/v1/retrieve/classify`| `RetrievalRequest`:<br>- `query` (str) | `ClassificationResult`:<br>- `category`<br>- `sub_category`<br>- `intent`<br>- `entities` (authors, topics, journals)<br>- `extracted_filters`<br>- `execution_plan` | Phân tích câu hỏi người dùng, xác định xem câu hỏi thuộc loại chitchat, tra cứu DOI, đếm số liệu, tra cứu đồ thị hay tìm kiếm vector. |

---

### Nhóm 5: Giai đoạn 4 - Sinh phản hồi căn cứ (Phase 4: Generation)
| Method | Endpoint | Request Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/generate` | `GenerationRequest`:<br>- `query` (str)<br>- `contexts` (List[ContextItem])<br>- `model` (str, optional)<br>- `temperature` (float, optional)<br>- `history_context` (str, optional) | `GenerationResponse`:<br>- `query`<br>- `answer`<br>- `model`<br>- `contexts_used`<br>- `citations`<br>- `usage` (tokens)<br>- `latency_ms` | Nhận danh sách ngữ cảnh đã có sẵn, định dạng prompt học thuật và gọi mô hình ngôn ngữ sinh câu trả lời có trích dẫn. |

---

### Nhóm 6: Pipeline Chat Hợp nhất (End-to-End RAG Pipeline)
| Method | Endpoint | Request Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/chat` | `RagPipelineRequest`:<br>- `query` (str, bắt buộc)<br>- `project_id` (int, optional)<br>- `user_id` (str, optional)<br>- `top_k` (int, default 5)<br>- `include_contexts` (bool, default true)<br>- `save_history` (bool, default true) | `RagPipelineResponse`:<br>- `query`<br>- `answer`<br>- `model`<br>- `contexts` (list)<br>- `citations` (list)<br>- `latency_breakdown` (retrieval_ms, generation_ms, total_ms)<br>- `user_message_id`<br>- `assistant_message_id` | **Endpoint chính của hệ thống**: Tích hợp giải quyết đại từ từ lượt hội thoại trước -> Truy xuất kết hợp -> Tạo prompt -> Gọi LLM -> Cập nhật bộ nhớ ngữ cảnh -> Ghi vết tin nhắn vào PostgreSQL. |

---

### Nhóm 7: Quản lý Lịch sử trò chuyện (Chat History)
| Method | Endpoint | Params / Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/chat/history` | Query params:<br>- `project_id` (int, opt)<br>- `user_id` (str, opt)<br>- `limit` (int, 1-100)<br>- `offset` (int)<br>- `order` ("asc" / "desc") | `ChatHistoryResponse`:<br>- `messages` (List[ChatMessageItem])<br>- `total_count`<br>- `project_id`<br>- `user_id` | Lấy danh sách lịch sử hội thoại có phân trang theo đề tài hoặc người dùng. |
| `POST` | `/api/v1/chat/messages` | `ChatMessageCreate`:<br>- `project_id`, `user_id`, `role`, `content`, `model`, `tokens`... | `ChatMessageItem` (HTTP 201) | Lưu thủ công một bản ghi tin nhắn vào cơ sở dữ liệu. |
| `GET` | `/api/v1/chat/messages/{message_id}` | Path param: `message_id` (int) | `ChatMessageItem` (HTTP 200) | Lấy chi tiết thông tin và số token của một tin nhắn cụ thể. |
| `DELETE`| `/api/v1/chat/messages/{message_id}`| Path param: `message_id` (int) | `DeleteHistoryResponse` (HTTP 200) | Xóa một tin nhắn theo ID. |
| `DELETE`| `/api/v1/chat/history` | Query params:<br>- `project_id` (int, opt)<br>- `user_id` (str, opt) | `DeleteHistoryResponse`:<br>- `deleted_count`<br>- `message` | Xóa sạch toàn bộ lịch sử trò chuyện của một đề tài hoặc một người dùng. |

---

### Nhóm 8: Quản lý Bộ nhớ làm việc tức thời (Context Memory)
| Method | Endpoint | Params / Body | Response | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/chat/context` | Query params:<br>- `project_id` (int, opt)<br>- `user_id` (str, opt) | `ContextMemoryState`:<br>- `active_topic`<br>- `active_year`<br>- `referenced_articles`<br>- `referenced_authors`<br>- `recent_turns` (List[DialogueTurn]) | Lấy trạng thái bộ nhớ hội thoại trượt đang hoạt động của phiên làm việc. |
| `POST` | `/api/v1/chat/context/update`| `UpdateContextMemoryRequest`:<br>- `active_topic`, `active_year`, `referenced_articles`... | `ContextMemoryState` | Can thiệp cập nhật thủ công chủ đề hoặc thực thể đang thảo luận trong phiên. |
| `POST` | `/api/v1/chat/context/reset` | Query params:<br>- `project_id`, `user_id` | `ResetContextMemoryResponse` | Xóa trắng bộ nhớ trượt của phiên để bắt đầu chủ đề thảo luận hoàn toàn mới. |

---

## 8. Hướng dẫn cài đặt & Khởi chạy

### Cách 1: Chạy trực tiếp qua Docker Compose (Khuyến nghị)
Hạ tầng docker sẽ khởi chạy RAG Application cùng cấu hình mạng kết nối tới PostgreSQL và Neo4j:

```bash
# 1. Khởi chạy toàn bộ hệ thống
docker-compose up --build -d

# 2. Xem logs thời gian thực
docker-compose logs -f rag-app

# 3. Kiểm tra trạng thái
curl http://localhost:8000/health
```

### Cách 2: Chạy cục bộ trên máy phát triển (Local Development)

#### Yêu cầu:
- Python 3.10 hoặc 3.11
- Công cụ quản lý Poetry (hoặc pip)
- Cài đặt Ollama (nếu dùng LLM cục bộ) và tải model: `ollama run llama3.2:3b`

#### Các bước cài đặt:
```bash
# 1. Cài đặt các thư viện phụ thuộc
poetry install

# 2. Cấu hình biến môi trường
cp .env.example .env
# Chỉnh sửa thông tin kết nối DB (Postgres, Neo4j, Gemini API key...) trong .env

# 3. Khởi chạy máy chủ FastAPI
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập Swagger UI tại: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 9. Đánh giá chất lượng RAG (Evaluation & Testing)

Hệ thống tích hợp sẵn bộ công cụ kiểm thử toàn diện:

### 1. Chạy Unit & Integration Tests:
```bash
poetry run pytest tests/ -v
```
Kiểm tra tính đúng đắn của API contracts, xử lý lỗi, phân loại câu hỏi và cơ chế cô lập phạm vi đề tài (Project Scope).

### 2. Chạy Evaluation Framework:
```bash
python eval/run_eval.py
```
Bộ đánh giá sẽ thực thi các tác vụ tại `eval/tasks/` và chấm điểm định lượng qua `eval/graders/`:
- **Context Hit Rate & Precision**: Tỷ lệ tìm thấy đúng bài báo mục tiêu.
- **Answer Faithfulness**: Đảm bảo câu trả lời không xuất hiện hiện tượng bịa đặt (hallucination), bám sát 100% ngữ cảnh được cấp.
- **Citation Attribution**: Tính chính xác của các nguồn dẫn chứng trong câu trả lời.
- **Latency Benchmark**: Thời gian phản hồi trung bình của từng pha trong hệ thống.
