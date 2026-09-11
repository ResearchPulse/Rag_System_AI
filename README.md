# Rag_System_AI — Scientific Journal Publication Trend Tracking RAG System

Hệ thống RAG (Retrieval-Augmented Generation) phục vụ theo dõi và phân tích xu hướng công bố các bài báo khoa học. Dự án được xây dựng theo kiến trúc **Modular Monolith** (Đơn khối dạng mô-đun), kết hợp ưu điểm của việc triển khai đơn giản, hiệu năng cao (không tốn độ trễ mạng giữa các service nội bộ) cùng tính phân tách ranh giới module rõ ràng, sẵn sàng mở rộng.

---

## 1. Kiến trúc Modular Monolith

```
Rag_System_AI/
├── app/
│   ├── main.py                     # Entry point FastAPI duy nhất (Port 8000)
│   ├── api/
│   │   ├── deps.py                 # Dependency Injection cung cấp services
│   │   └── v1/
│   │       ├── router.py           # Gom tất cả router v1
│   │       └── endpoints/          # Endpoints theo từng giai đoạn RAG
│   │           ├── ingestion.py    # POST /api/v1/documents
│   │           ├── embedding.py    # POST /api/v1/embeddings
│   │           ├── retrieval.py    # POST /api/v1/retrieve
│   │           ├── generation.py   # POST /api/v1/generate
│   │           └── rag.py          # POST /api/v1/chat (End-to-End Pipeline)
│   ├── core/
│   │   ├── config.py               # Quản lý cấu hình qua pydantic-settings
│   │   └── logger.py               # Structured logging
│   ├── modules/                    # 4 Giai đoạn cốt lõi (12 bước RAG)
│   │   ├── ingestion/              # [Phase 1: Ingestion]
│   │   │   ├── data_sources/       # Step 1: Data Sources (PDF, API, Web, Transcripts)
│   │   │   ├── loaders/            # Step 2: Document Loading
│   │   │   ├── chunking/           # Step 3: Meaningful Chunking
│   │   │   ├── metadata/           # Step 4: Metadata Extraction
│   │   │   ├── schemas.py
│   │   │   └── service.py
│   │   ├── indexing/               # [Phase 2: Indexing]
│   │   │   ├── embedders/          # Step 5: Embeddings
│   │   │   ├── vector_store/       # Step 6: Vector Database
│   │   │   ├── schemas.py
│   │   │   └── service.py
│   │   ├── retrieval/              # [Phase 3: Retrieval]
│   │   │   ├── query_rewriting/    # Step 7: Query Rewriting
│   │   │   ├── hybrid_search/      # Step 8: Hybrid Search (Dense + Sparse BM25)
│   │   │   ├── reranking/          # Step 9: Cross-Encoder Reranking
│   │   │   ├── schemas.py
│   │   │   └── service.py
│   │   └── generation/             # [Phase 4: Generation & Evaluation]
│   │       ├── context_assembly/   # Step 10: Context Assembly
│   │       ├── llm/                # Step 11: LLM Generation
│   │       ├── evaluation/         # Step 12: Evaluation (Faithfulness, Cost, Citations)
│   │       ├── schemas.py
│   │       └── service.py
│   └── shared/                     # Shared models & Pipeline definitions
│       ├── schemas.py              # BaseResponse, ErrorResponse, HealthCheck
│       └── pipeline.py             # Định nghĩa chi tiết 12 bước (rag_steps)
├── scripts/
│   └── export_openapi.py           # Xuất hợp đồng openapi.json
├── docs/
│   └── openapi/
│       └── openapi.json            # File đặc tả OpenAPI JSON hợp nhất
├── tests/
│   └── test_api.py                 # Bộ kiểm thử tích hợp (Unit & Integration tests)
├── Dockerfile                      # Single-stage container image
├── docker-compose.yml              # Triển khai app + Vector DB (Qdrant)
├── pyproject.toml                  # Quản lý dependency tập trung qua Poetry
├── .env.example                    # Biến môi trường mẫu
├── .env                            # Biến môi trường local
├── .gitignore
└── README.md
```

---

## 2. Chu trình RAG Pipeline chuẩn hóa (12 Bước qua 4 Giai đoạn)

Hệ thống tổ chức ranh giới code nghiêm ngặt nhằm giải quyết 4 điểm lỗi chí mạng (*Four Failure Points*) thường gặp trong triển khai RAG:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION ("Garbage in, garbage out - Tránh chunk vội vã, bỏ sót metadata")    │
│  ├── [Step 1: Data Sources]         -> app/modules/ingestion/data_sources/        │
│  ├── [Step 2: Document Loading]     -> app/modules/ingestion/loaders/             │
│  ├── [Step 3: Meaningful Chunking]  -> app/modules/ingestion/chunking/            │
│  └── [Step 4: Metadata Extraction]  -> app/modules/ingestion/metadata/            │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 2. INDEXING ("Building the Brain - Xây dựng bộ nhớ vector hiệu năng cao")         │
│  ├── [Step 5: Embeddings]           -> app/modules/indexing/embedders/            │
│  └── [Step 6: Vector Database]      -> app/modules/indexing/vector_store/         │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 3. RETRIEVAL ("The Make-or-Break Stage - Tránh trả về số lượng thay vì chất lượng")│
│  ├── [Step 7: Query Rewriting]      -> app/modules/retrieval/query_rewriting/     │
│  ├── [Step 8: Hybrid Search]        -> app/modules/retrieval/hybrid_search/       │
│  └── [Step 9: Reranking]            -> app/modules/retrieval/reranking/           │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 4. GENERATION & EVALUATION ("Close the Loop - Đảm bảo tin cậy, có trích dẫn")    │
│  ├── [Step 10: Context Assembly]    -> app/modules/generation/context_assembly/   │
│  ├── [Step 11: LLM Generation]      -> app/modules/generation/llm/                │
│  └── [Step 12: Evaluation]          -> app/modules/generation/evaluation/         │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
                                 [Grounded Answer]
                  (Relevant, grounded, cited, fast, trustworthy)
```

---

## 3. Swagger UI & Danh mục API Endpoints

Toàn bộ hệ thống được expose tập trung tại một cổng duy nhất:
- **Base URL:** `http://localhost:8000`
- **Swagger UI Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON Spec:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

| Nhóm chức năng | Method | Endpoint | Mô tả |
| :--- | :--- | :--- | :--- |
| **System** | `GET` | `/health` | Kiểm tra trạng thái ứng dụng |
| | `GET` | `/` | Thông tin chào mừng & sitemap |
| **Phase 1: Ingestion** | `POST` | `/api/v1/documents` | Upload và phân tách chunking tài liệu khoa học |
| | `GET` | `/api/v1/documents/{id}` | Lấy chi tiết thông tin tài liệu và danh sách chunks |
| **Phase 2: Indexing** | `POST` | `/api/v1/embeddings` | Chuyển đổi chuỗi văn bản thành dense vector embeddings |
| **Phase 3: Retrieval** | `POST` | `/api/v1/retrieve` | Semantic vector search, hybrid search và cross-encoder rerank |
| **Phase 4: Generation** | `POST` | `/api/v1/generate` | Gọi LLM sinh câu trả lời căn cứ chặt chẽ vào context |
| **End-to-End RAG** | `POST` | `/api/v1/chat` | Luồng Chat RAG hợp nhất (Retrieve -> Assembly -> Generate) |

---

## 4. Hướng dẫn khởi chạy

### Cách 1: Chạy bằng Docker Compose (Khuyến nghị)
Bao gồm ứng dụng Modular Monolith và cơ sở dữ liệu Vector Qdrant:

```bash
# Khởi chạy ứng dụng và Vector DB
docker-compose up --build

# Hoặc chạy nền:
docker-compose up -d --build
```
Truy cập:
- RAG Application: [http://localhost:8000/docs](http://localhost:8000/docs)
- Qdrant Vector DB Web Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

Dừng hệ thống:
```bash
docker-compose down
```

---

### Cách 2: Chạy trực tiếp trên máy cục bộ (Local Development)

#### Yêu cầu:
- Python 3.11+
- Poetry hoặc pip

#### Cài đặt và khởi chạy:
```bash
# 1. Cài đặt dependency
poetry install
# Hoặc: pip install fastapi uvicorn pydantic pydantic-settings python-multipart httpx

# 2. Khởi chạy server Uvicorn
poetry run uvicorn app.main:app --reload --port 8000
# Hoặc:
# uvicorn app.main:app --reload --port 8000
```

---

## 5. Kiểm thử & Xuất đặc tả OpenAPI

### Chạy Unit Test:
```bash
python -m unittest discover tests
```

### Xuất lại OpenAPI Spec:
```bash
python scripts/export_openapi.py
```
File đặc tả sẽ được cập nhật tự động tại: `docs/openapi/openapi.json`.
