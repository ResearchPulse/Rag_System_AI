# Rag_System_AI — Scientific Journal Publication Trend Tracking RAG System

Hệ thống RAG (Retrieval-Augmented Generation) phục vụ theo dõi và phân tích xu hướng công bố các bài báo khoa học. Dự án được thiết kế theo kiến trúc **Monorepo Microservices**, trong đó mỗi microservice là một ứng dụng FastAPI độc lập, quản lý dependency bằng Poetry, có thể chạy riêng lẻ bằng `uvicorn` hoặc triển khai đồng bộ qua Docker Compose.

---

## 1. Kiến trúc tổng thể

```
Rag_System_AI/
├── services/
│   ├── ingestion-service/      # Port 8001: Nhận upload tài liệu, chunking, lưu metadata
│   ├── embedding-service/      # Port 8002: Chuyển text chunks thành dense vector embeddings
│   ├── retrieval-service/      # Port 8003: Tìm kiếm vector tương đồng, rerank top-k context
│   ├── generation-service/     # Port 8004: Gọi LLM sinh câu trả lời có trích dẫn context
│   └── gateway-service/        # Port 8005: API Gateway duy nhất expose cho client
├── shared/
│   ├── schemas/                # Pydantic schemas dùng chung giữa các service
│   └── utils/                  # Utility & Logger dùng chung
├── docs/
│   └── openapi/                # Lưu file OpenAPI JSON contracts đã trích xuất
├── docker-compose.yml          # Điều phối 5 services trên dải port 8001-8005
├── .gitignore
└── README.md
```

### Quy trình dữ liệu (Pipeline flow):
```
Client  ───> [gateway-service :8005]
                     │
                     ├─(1. Query)──> [retrieval-service :8003] ──(Embed Query)──> [embedding-service :8002]
                     │                       │
                     │                 (Top-K Chunks)
                     │                       │
                     └─(2. Query + Chunks)──> [generation-service :8004] ──(LLM)
                                                     │
Client  <─── (Final Answer + Citations) ─────────────┘
```

---

## 2. Chu trình RAG Pipeline chuẩn hóa (12 Steps across 4 Phases)

Hệ thống chia nhỏ toàn bộ luồng RAG thành **12 bước** tương ứng trong các microservices, kiểm soát nghiêm ngặt 4 điểm lỗi thường gặp (*Failure Points*):

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION ("Garbage in, garbage out - Tránh chunk vội vã, bỏ qua metadata")    │
│  [Step 1: Data Sources]         -> Thu thập PDF, APIs, Web, Transcripts           │
│  [Step 2: Document Loading]     -> Parse và chuẩn hóa tài liệu đa định dạng       │
│  [Step 3: Meaningful Chunking]  -> Phân tách theo ngữ nghĩa, bảo toàn văn cảnh   │
│  [Step 4: Metadata Extraction]  -> Trích xuất tiêu đề, tác giả, năm, quartile...  │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 2. INDEXING ("Building the Brain - Xây dựng bộ nhớ vector hiệu năng cao")         │
│  [Step 5: Embeddings]           -> Chuyển đổi chunks thành high-dimensional vector │
│  [Step 6: Vector Database]      -> Lưu trữ và lập chỉ mục (Indexed Knowledge)     │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 3. RETRIEVAL ("The Make-or-Break Stage - Tránh trả về số lượng thay vì chất lượng")│
│  [Step 7: Query Rewriting]      -> Làm rõ ý định người dùng, mở rộng query       │
│  [Step 8: Hybrid Search]        -> Kết hợp Vector Dense + Lexical Sparse (BM25)   │
│  [Step 9: Reranking]            -> Rerank bằng Cross-Encoder chấm điểm chính xác  │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 4. GENERATION & EVALUATION ("Close the Loop - Đảm bảo tin cậy, có trích dẫn")    │
│  [Step 10: Context Assembly]    -> Chọn lọc, sắp xếp và nén ngữ cảnh tối ưu       │
│  [Step 11: LLM Generation]      -> Sinh câu trả lời căn cứ chặt chẽ vào context   │
│  [Step 12: Evaluation]          -> Đo lường Faithfulness, Latency, Cost, Citation │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
                                 [Grounded Answer]
                  (Relevant, grounded, cited, fast, trustworthy)
```

---

## 3. Bảng danh mục Microservices & Swagger UI


| Service | Port | Swagger UI (Docs) | OpenAPI Spec | Mô tả chính |
| :--- | :--- | :--- | :--- | :--- |
| **ingestion-service** | `8001` | [http://localhost:8001/docs](http://localhost:8001/docs) | `/openapi.json` | Upload tài liệu (PDF, TXT, DOCX), phân tách chunking, quản lý metadata |
| **embedding-service** | `8002` | [http://localhost:8002/docs](http://localhost:8002/docs) | `/openapi.json` | Nhận danh sách text chunks, sinh vector float biểu diễn |
| **retrieval-service** | `8003` | [http://localhost:8003/docs](http://localhost:8003/docs) | `/openapi.json` | Tìm kiếm ngữ cảnh vector tương đồng, reranking top-k context |
| **generation-service**| `8004` | [http://localhost:8004/docs](http://localhost:8004/docs) | `/openapi.json` | Nhận query và context, gọi LLM tạo câu trả lời chuẩn xác kèm trích dẫn |
| **gateway-service**   | `8005` | [http://localhost:8005/docs](http://localhost:8005/docs) | `/openapi.json` | Cổng API duy nhất cho Client, điều phối chuỗi RAG Pipeline |

---

## 3. Danh sách Endpoints chi tiết

### 3.1. Ingestion Service (`:8001`)
- `POST /api/v1/documents` — Tải lên tài liệu hoặc body text, thực hiện chunking và trả về danh sách chunks.
- `GET /api/v1/documents/{id}` — Lấy chi tiết tài liệu, trạng thái và các chunks đã tạo.
- `GET /health` — Kiểm tra trạng thái service.

### 3.2. Embedding Service (`:8002`)
- `POST /api/v1/embeddings` — Nhận mảng chuỗi văn bản `texts`, trả về mảng vector embeddings và số token tiêu thụ.
- `GET /health` — Kiểm tra trạng thái service.

### 3.3. Retrieval Service (`:8003`)
- `POST /api/v1/retrieve` — Nhận query, `top_k`, `score_threshold`, `rerank`, trả về top context passages liên quan nhất.
- `GET /health` — Kiểm tra trạng thái service.

### 3.4. Generation Service (`:8004`)
- `POST /api/v1/generate` — Nhận query và danh sách contexts, gọi mô hình LLM để sinh câu trả lời trích dẫn.
- `GET /health` — Kiểm tra trạng thái service.

### 3.5. Gateway Service (`:8005`)
- `POST /api/v1/chat` — Điểm tiếp nhận câu hỏi từ Client, điều phối tìm kiếm ngữ cảnh và tổng hợp câu trả lời hoàn chỉnh.
- `GET /health` — Kiểm tra trạng thái service.

---

## 4. Hướng dẫn chạy dự án

### Cách 1: Chạy toàn bộ hệ thống bằng Docker Compose (Khuyến nghị)

Yêu cầu: Đã cài đặt [Docker](https://www.docker.com/) và Docker Compose.

```bash
# 1. Clone hoặc mở thư mục gốc của dự án
cd Rag_System_AI

# 2. Khởi chạy toàn bộ 5 microservices
docker-compose up --build

# Hoặc chạy nền:
docker-compose up -d --build
```

Sau khi khởi động thành công, bạn có thể truy cập Swagger UI của từng service theo bảng danh mục ở mục 2.

Để dừng toàn bộ:
```bash
docker-compose down
```

---

### Cách 2: Chạy từng Service riêng lẻ bằng Uvicorn

Mỗi service hoàn toàn độc lập và có thể chạy trực tiếp bằng `uvicorn` trên máy phát triển.

#### Yêu cầu môi trường:
- Python 3.11+
- Poetry hoặc pip

#### Cài đặt và khởi chạy:

**1. Ingestion Service (Port 8001)**
```bash
cd services/ingestion-service
poetry install
poetry run uvicorn app.main:app --reload --port 8001
# Hoặc với python thông thường:
# uvicorn app.main:app --reload --port 8001
```

**2. Embedding Service (Port 8002)**
```bash
cd services/embedding-service
poetry install
poetry run uvicorn app.main:app --reload --port 8002
```

**3. Retrieval Service (Port 8003)**
```bash
cd services/retrieval-service
poetry install
poetry run uvicorn app.main:app --reload --port 8003
```

**4. Generation Service (Port 8004)**
```bash
cd services/generation-service
poetry install
poetry run uvicorn app.main:app --reload --port 8004
```

**5. Gateway Service (Port 8005)**
```bash
cd services/gateway-service
poetry install
poetry run uvicorn app.main:app --reload --port 8005
```

---

## 5. Trích xuất hợp đồng OpenAPI Specification

Mỗi service được trang bị script `scripts/export_openapi.py` giúp tự động trích xuất định nghĩa API (`openapi.json`) ra thư mục `docs/openapi/<service-name>.json` nhằm phục vụ việc review hợp đồng API (API contract review) và tích hợp client SDK:

```bash
# Export cho Ingestion Service
python services/ingestion-service/scripts/export_openapi.py

# Export cho Embedding Service
python services/embedding-service/scripts/export_openapi.py

# Export cho Retrieval Service
python services/retrieval-service/scripts/export_openapi.py

# Export cho Generation Service
python services/generation-service/scripts/export_openapi.py

# Export cho Gateway Service
python services/gateway-service/scripts/export_openapi.py
```

Các file JSON sau khi export sẽ nằm tại thư mục `docs/openapi/`:
- `docs/openapi/ingestion-service.json`
- `docs/openapi/embedding-service.json`
- `docs/openapi/retrieval-service.json`
- `docs/openapi/generation-service.json`
- `docs/openapi/gateway-service.json`

---

## 6. Cấu trúc chuẩn của một Service

Mọi service trong thư mục `services/` đều tuân thủ chặt chẽ pattern sau:

```
<service-name>/
├── app/
│   ├── main.py              # Khởi tạo FastAPI với title, description, version, health check
│   ├── api/
│   │   └── v1/
│   │       └── routes.py    # Định nghĩa endpoint API, prefix /api/v1
│   ├── schemas/             # Pydantic Request/Response models với đầy đủ example
│   ├── services/            # Business logic (được tách lớp độc lập với API handler)
│   └── core/
│       └── config.py        # Đọc biến môi trường bằng pydantic-settings
├── scripts/
│   └── export_openapi.py    # Script trích xuất OpenAPI JSON contract
├── pyproject.toml           # Quản lý dependencies qua Poetry
├── Dockerfile               # Container build script (python:3.11-slim)
├── .env.example             # Mẫu cấu hình môi trường
├── .env                     # Biến môi trường cục bộ
└── README.md                # Tài liệu hướng dẫn riêng của service
```
