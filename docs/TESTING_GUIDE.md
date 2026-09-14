# HƯỚNG DẪN TỪNG BƯỚC KIỂM THỬ TOÀN BỘ HỆ THỐNG RAG_SYSTEM_AI
## (Comprehensive Step-by-Step Testing Guide & Test Cases Matrix)

> **Tài liệu dự án**: Scientific Journal Publication Trend Tracking System (`ResearchPulse`)  
> **Phiên bản**: 0.1.0  
> **Kiến trúc**: FastAPI Modular Monolith (4 Phases - 12 Steps RAG + Extended Modules)  
> **Branch**: `feature/implement-apis`  
> **Swagger UI URL**: [http://localhost:8000/docs](http://localhost:8000/docs)  
> **Redoc URL**: [http://localhost:8000/redoc](http://localhost:8000/redoc)  

---

## MỤC LỤC

1. [Tổng Quan Hệ Thống & Phạm Vi Kiểm Thử](#1-tổng-quan-hệ-thống--phạm-vi-kiểm-thử)
2. [Chuẩn Bị Môi Trường & Khởi Động Server](#2-chuẩn-bị-môi-trường--khởi-động-server)
3. [Hướng Dẫn Chạy Test Tự Động (Automated Testing Suite)](#3-hướng-dẫn-chạy-test-tự-động-automated-testing-suite)
4. [Hướng Dẫn Từng Bước Kiểm Thử Thủ Công (Manual Testing Guide)](#4-hướng-dẫn-từng-bước-kiểm-thử-thủ-công-manual-testing-guide)
   - [4.1 System & Health Endpoints](#41-system--health-endpoints)
   - [4.2 Phase 1: Ingestion & Document Processing](#42-phase-1-ingestion--document-processing)
   - [4.3 Phase 2: Indexing & Vector Embeddings](#43-phase-2-indexing--vector-embeddings)
   - [4.4 Phase 3: Query Classification & Retrieval](#44-phase-3-query-classification--retrieval)
   - [4.5 Phase 4: Generation & LLM Synthesis](#45-phase-4-generation--llm-synthesis)
   - [4.6 End-to-End Chatbot RAG Pipeline](#46-end-to-end-chatbot-rag-pipeline)
   - [4.7 Chức năng 1: Lưu trữ Lịch sử Trò chuyện (Chat History API)](#47-chức-năng-1-lưu-trữ-lịch-sử-trò-chuyện-chat-history-api)
   - [4.8 Chức năng 2: Nhận diện Phạm vi Đề tài (Project Scope)](#48-chức-năng-2-nhận-diện-phạm-vi-đề-tài-project-scope)
   - [4.9 Chức năng 3: Quản lý Ngữ cảnh Hội thoại (Context Memory)](#49-chức-năng-3-quản-lý-ngữ-cảnh-hội-thoại-context-memory)
5. [Bảng Toàn Bộ Test Cases (All Test Cases Matrix)](#5-bảng-toàn-bộ-test-cases-all-test-cases-matrix)
6. [Kiểm Thử Khả Năng Chịu Lỗi & Ngoại Lệ (Fault Tolerance & Edge Cases)](#6-kiểm-thử-khả-năng-chịu-lỗi--ngoại-lệ-fault-tolerance--edge-cases)

---

## 1. TỔNG QUAN HỆ THỐNG & PHẠM VI KIỂM THỬ

Hệ thống `Rag_System_AI` được thiết kế theo mô hình **Modular Monolith** hoàn chỉnh, đảm nhiệm toàn bộ quy trình RAG học thuật và các dịch vụ bổ trợ:

```
[Client / Frontend / Swagger UI]
               │
               ▼
   [FastAPI Modular Monolith (Port 8000)]
   ├── System Endpoints: GET /health, GET /
   ├── Phase 1: Ingestion (Documents, Semantic Chunking, Metadata)
   ├── Phase 2: Indexing & Embeddings (bge-base-en-v1.5, pgvector)
   ├── Phase 3: Retrieval (Query Classifier 8 Scenarios, BM25 + Vector + Graph Neo4j, Reranker)
   ├── Phase 4: Generation (Context Assembly, LLM / Ollama / DeepSeek Synthesis)
   ├── End-to-End Chat: POST /api/v1/chat
   ├── Feature 1: Chat History (CRUD Lịch sử trò chuyện PostgreSQL "Project_Chat_Message")
   ├── Feature 2: Project Scope Recognition (Giới hạn truy xuất & phân tích theo project_id)
   └── Feature 3: Context Memory (Bộ nhớ ngữ cảnh ngắn hạn, Working Entities, Coreference Reformulation)
```

Tất cả các module đều sở hữu cơ chế **Resilient Fallback** (tự động chuyển sang bộ nhớ đệm / in-memory / extractive guardrail nếu PostgreSQL, Neo4j hoặc Ollama chưa sẵn sàng trên máy dev), đảm bảo 100% API luôn hoạt động mà không bị crash.

---

## 2. CHUẨN BỊ MÔI TRƯỜNG & KHỞI ĐỘNG SERVER

### Bước 2.1: Cấu hình biến môi trường (`.env`)
Đảm bảo file `.env` tại thư mục gốc `Rag_System_AI/` có cấu hình cơ bản:
```env
APP_NAME=Rag_System_AI
APP_ENV=development
API_PORT=8000

# PostgreSQL (Chứa bài báo, vector embedding, chat history, project scope)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=researchpulse
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Neo4j (Knowledge Graph quan hệ tác giả, đề tài, tạp chí)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Ollama LLM (Mô hình sinh câu trả lời)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

### Bước 2.2: Khởi động FastAPI Server
Mở terminal tại thư mục `Rag_System_AI`:
```powershell
poetry run uvicorn app.main:app --reload --port 8000
```
Khi hiển thị log:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```
Server đã sẵn sàng tiếp nhận request.

---

## 3. HƯỚNG DẪN CHẠY TEST TỰ ĐỘNG (AUTOMATED TESTING SUITE)

Hệ thống được tích hợp bộ kiểm thử tự động toàn diện với **22 Unit & Integration Tests**.

### Lệnh chạy toàn bộ 22 Tests:
```powershell
poetry run pytest -v
```

### Kết quả chuẩn khi vượt qua toàn bộ test:
```
tests/test_api.py::TestModularMonolithAPI::test_chat_pipeline PASSED
tests/test_api.py::TestModularMonolithAPI::test_health_check PASSED
tests/test_api.py::TestModularMonolithAPI::test_root PASSED
tests/test_chat_history.py::TestChatHistoryAPI::test_chat_pipeline_auto_record_history PASSED
tests/test_chat_history.py::TestChatHistoryAPI::test_clear_chat_history PASSED
tests/test_chat_history.py::TestChatHistoryAPI::test_create_and_get_chat_message PASSED
tests/test_chat_history.py::TestChatHistoryAPI::test_get_chat_history_pagination PASSED
tests/test_context_memory.py::TestContextMemory::test_api_context_endpoints PASSED
tests/test_context_memory.py::TestContextMemory::test_format_history_for_prompt PASSED
tests/test_context_memory.py::TestContextMemory::test_record_turn_and_memory_state PASSED
tests/test_project_scope.py::TestProjectScopeRAG::test_chat_pipeline_with_project_scope PASSED
tests/test_project_scope.py::TestProjectScopeRAG::test_project_scope_metadata PASSED
tests/test_project_scope.py::TestProjectScopeRAG::test_retrieval_with_project_id PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_1_sql_aggregation PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_2_semantic_similarity PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_3_metadata_lookup_doi PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_4_co_authorship PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_6_hybrid_filtered_graph PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_7_chitchat PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_scenario_8_clarification_needed PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_service_dispatches_graph_traversal PASSED
tests/test_query_classifier.py::TestQueryClassifierStandard::test_service_dispatches_sql_aggregation PASSED

================== 22 passed, 1 warning in 71.32s ===================
```

### Chạy từng file test riêng biệt:
```powershell
# 1. Test cấu trúc API Monolith & Health
poetry run pytest tests/test_api.py -v

# 2. Test Quản lý Lịch sử Chat (Chat History)
poetry run pytest tests/test_chat_history.py -v

# 3. Test Giới hạn Phạm vi Đề tài (Project Scope Recognition)
poetry run pytest tests/test_project_scope.py -v

# 4. Test Bộ nhớ Ngữ cảnh & Đa lượt hội thoại (Context Memory)
poetry run pytest tests/test_context_memory.py -v

# 5. Test Bộ phân loại câu hỏi (Query Classifier 8 Scenarios)
poetry run pytest tests/test_query_classifier.py -v
```

---

## 4. HƯỚNG DẪN TỪNG BƯỚC KIỂM THỬ THỦ CÔNG (MANUAL TESTING GUIDE)

Bạn có thể kiểm thử trực tiếp trên trình duyệt bằng cách mở **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) hoặc sử dụng terminal cURL / Postman theo các bước chi tiết sau:

---

### 4.1 System & Health Endpoints

#### Bước 1: Kiểm tra Service Health
- **Endpoint**: `GET /health`
- **Mục tiêu**: Xác nhận server đang chạy, hiển thị đúng phiên bản và môi trường.
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/health"
  ```
- **Kết quả mong đợi (HTTP 200)**:
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "environment": "development"
  }
  ```

#### Bước 2: Kiểm tra Welcome Root & Sitemap
- **Endpoint**: `GET /`
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/"
  ```
- **Kết quả mong đợi (HTTP 200)**: Chứa thông tin sitemap API và kiến trúc `Modular Monolith`.

---

### 4.2 Phase 1: Ingestion & Document Processing

#### Bước 3: Nạp và Semantic Chunking một bài báo khoa học
- **Endpoint**: `POST /api/v1/documents`
- **Mục tiêu**: Tách văn bản bài báo thành các đoạn semantic chunks và trích xuất metadata (tác giả, năm, DOI).
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/documents" \
    -H "Content-Type: application/json" \
    -d '{
      "title": "A Survey on Graph Retrieval-Augmented Generation in Scholarly Systems",
      "content": "Graph Retrieval-Augmented Generation (Graph RAG) combines structured knowledge graphs with unstructured text embeddings. In 2026, researchers demonstrated a 34% improvement in multi-hop reasoning over academic citation networks.",
      "authors": ["Dr. Alan Turing", "Prof. Geoffrey Hinton"],
      "year": 2026,
      "doi": "10.1016/j.graphrag.2026.01.001",
      "subject_category": "Artificial Intelligence"
    }'
  ```
- **Kết quả mong đợi (HTTP 201 Created)**:
  - Trả về `document_id` (ví dụ: `doc_...`), `total_chunks >= 1`, và danh sách các chunks.

#### Bước 4: Xem chi tiết tài liệu đã nạp
- **Endpoint**: `GET /api/v1/documents/{document_id}`
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/documents/doc_demo_sample"
  ```
- **Kết quả mong đợi (HTTP 200)**: Trả về đầy đủ thông tin bài báo và danh sách chunks.

---

### 4.3 Phase 2: Indexing & Vector Embeddings

#### Bước 5: Sinh Vector Embedding cho các đoạn văn bản
- **Endpoint**: `POST /api/v1/embeddings`
- **Mục tiêu**: Mã hóa văn bản thành mảng vector số thực 768 chiều (bge-base-en-v1.5).
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
      "texts": [
        "Graph RAG significantly improves scholarly trend analysis.",
        "Deep learning models require high-quality citation data."
      ],
      "model": "BAAI/bge-base-en-v1.5"
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - `embeddings`: Mảng 2 vector.
  - `dimension`: `768`.
  - `total_tokens`: Số lượng token tương ứng.

---

### 4.4 Phase 3: Query Classification & Retrieval

#### Bước 6: Phân loại câu hỏi tự động (Query Classification Layer)
- **Endpoint**: `POST /api/v1/retrieve/classify`
- **Mục tiêu**: Nhận diện ý định câu hỏi để quyết định đi qua SQL, Vector, Đồ thị Neo4j hay Chitchat.
- **Kịch bản A: Câu hỏi đếm / thống kê SQL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/retrieve/classify" \
    -H "Content-Type: application/json" \
    -d '{"query": "Có bao nhiêu bài báo xuất bản năm 2024?"}'
  ```
  - **Kết quả**: `category`: `"direct_lookup"`, `sub_category`: `"sql_aggregation"`, `requires_sql_aggregation`: `true`.
- **Kịch bản B: Câu hỏi quan hệ tác giả (Graph Traversal)**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/retrieve/classify" \
    -H "Content-Type: application/json" \
    -d '{"query": "Những ai là đồng tác giả của GS. Nguyễn Văn A?"}'
  ```
  - **Kết quả**: `category`: `"relational_reasoning"`, `sub_category`: `"co_authorship"`, `requires_graph_traversal`: `true`.
- **Kịch bản C: Chào hỏi xã giao (Chitchat Guardrail)**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/retrieve/classify" \
    -H "Content-Type: application/json" \
    -d '{"query": "Xin chào bạn, bạn là ai?"}'
  ```
  - **Kết quả**: `category`: `"chitchat"`, `requires_vector_search`: `false`.

#### Bước 7: Thực thi Truy xuất Kết hợp & Reranking (Hybrid Retrieval)
- **Endpoint**: `POST /api/v1/retrieve`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/retrieve" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Deep learning trends in medical image processing",
      "top_k": 3
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**: Trả về danh sách `results` gồm các chunk phù hợp nhất, kèm theo `score` và `rerank_score`.

---

### 4.5 Phase 4: Generation & LLM Synthesis

#### Bước 8: Sinh câu trả lời có kiểm chứng nguồn (Grounded Generation)
- **Endpoint**: `POST /api/v1/generate`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Lợi ích nổi bật của Graph RAG là gì?",
      "contexts": [
        "Graph RAG kết hợp cấu trúc tri thức quan hệ với vector embedding, giúp giảm hiện tượng ảo giác (hallucination) 45% và cải thiện khả năng suy luận đa bước qua mạng lưới trích dẫn."
      ],
      "model": "llama3.2:3b"
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**: Trả về `answer` được tổng hợp rõ ràng từ ngữ cảnh đã cấp, có trích dẫn nguồn.

---

### 4.6 End-to-End Chatbot RAG Pipeline

#### Bước 9: Trò chuyện toàn diện qua Pipeline RAG
- **Endpoint**: `POST /api/v1/chat`
- **Mục tiêu**: Điều phối tự động từ phân loại -> truy xuất -> tái xếp hạng -> tổng hợp ngữ cảnh -> sinh câu trả lời.
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Tổng quan các xu hướng nghiên cứu AI nổi bật năm 2026",
      "model": "llama3.2:3b"
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - `query`: Câu hỏi ban đầu.
  - `answer`: Câu trả lời hoàn chỉnh từ AI.
  - `retrieved_contexts`: Các đoạn trích dẫn được sử dụng.
  - `intent_category`: Thể loại câu hỏi nhận diện được.

---

### 4.7 Chức năng 1: Lưu trữ Lịch sử Trò chuyện (Chat History API)

#### Bước 10: Tự động ghi lại lịch sử khi Chatbot trả lời
- **Endpoint**: `POST /api/v1/chat`
- **Dữ liệu**: Truyền thêm `user_id` và `save_history: true`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Cho tôi biết các công trình nổi bật của tác giả Geoffrey Hinton",
      "project_id": 10,
      "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
      "save_history": true
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - Trả về `user_message_id` (ID tin nhắn câu hỏi của User).
  - Trả về `assistant_message_id` (ID tin nhắn phản hồi của Bot).

#### Bước 11: Lấy danh sách lịch sử trò chuyện có phân trang
- **Endpoint**: `GET /api/v1/chat/history?project_id=10&user_id=a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d&limit=10&order=asc`
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/chat/history?project_id=10&limit=10&order=asc"
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - `total`: Tổng số tin nhắn.
  - `messages`: Mảng các tin nhắn được sắp xếp theo thời gian, gồm `role` (`USER`/`ASSISTANT`), `content`, `created_at`.

#### Bước 12: Tạo tin nhắn thủ công (Manual message insertion)
- **Endpoint**: `POST /api/v1/chat/messages`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat/messages" \
    -H "Content-Type: application/json" \
    -d '{
      "project_id": 10,
      "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
      "role": "USER",
      "content": "Ghi chú bổ sung cho đề tài nghiên cứu"
    }'
  ```
- **Kết quả mong đợi (HTTP 201 Created)**: Trả về chi tiết tin nhắn kèm `message_id`.

#### Bước 13: Lấy chi tiết một tin nhắn theo ID
- **Endpoint**: `GET /api/v1/chat/messages/{message_id}`
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/chat/messages/1"
  ```
- **Kết quả mong đợi (HTTP 200)**: Chi tiết tin nhắn hoặc `404 Not Found` nếu ID không tồn tại.

#### Bước 14: Xóa một tin nhắn cụ thể
- **Endpoint**: `DELETE /api/v1/chat/messages/{message_id}`
- **cURL**:
  ```bash
  curl -X DELETE "http://localhost:8000/api/v1/chat/messages/1"
  ```
- **Kết quả mong đợi (HTTP 200)**: `{"deleted_count": 1, "message": "Deleted successfully."}`.

#### Bước 15: Xóa toàn bộ lịch sử trò chuyện của Project / User
- **Endpoint**: `DELETE /api/v1/chat/history?project_id=10`
- **cURL**:
  ```bash
  curl -X DELETE "http://localhost:8000/api/v1/chat/history?project_id=10"
  ```
- **Kết quả mong đợi (HTTP 200)**: `{"deleted_count": N, "message": "All chat history cleared successfully."}`.

---

### 4.8 Chức năng 2: Nhận diện Phạm vi Đề tài (Project Scope)

#### Bước 16: Test truy vấn RAG trong phạm vi một Project cụ thể
- **Endpoint**: `POST /api/v1/chat`
- **Mục tiêu**: Giới hạn toàn bộ việc tìm kiếm bài báo, tác giả và thống kê trong phạm vi chuyên môn của `project_id` (trang `trending_page`).
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Tổng quan các bài báo và tác giả hàng đầu trong đề tài này",
      "project_id": 12,
      "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - Trong `retrieved_contexts`, xuất hiện block đầu tiên dạng:
    ```markdown
    ### 🎯 Phạm vi Đề tài Nghiên cứu của Dự án (Project Scope #12)
    - Tên Đề tài / Dự án: ...
    - Lĩnh vực Chuyên môn: ...
    - Từ khóa Trọng tâm: ...
    - Quy mô Kho Tri thức Thuộc Phạm vi: ... bài báo khoa học đã được chọn lọc
    ```
  - Câu trả lời của Chatbot tập trung chính xác vào nội dung của đề tài, không bị lan man sang các ngành khác.

#### Bước 17: Test truy xuất Retrieval lọc theo Project Scope
- **Endpoint**: `POST /api/v1/retrieve`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/retrieve" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Các phương pháp học tăng cường tiên tiến",
      "project_id": 12,
      "top_k": 3
    }'
  ```
- **Kết quả mong đợi**: Kết quả truy xuất chứa chunk metadata `{"type": "project_scope", "project_id": 12}` và các bài báo nằm trong danh mục/từ khóa của đề tài 12.

---

### 4.9 Chức năng 3: Quản lý Ngữ cảnh Hội thoại (Context Memory)

#### Bước 18: Lấy trạng thái bộ nhớ ngữ cảnh hiện tại
- **Endpoint**: `GET /api/v1/chat/context?project_id=12&user_id=a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d`
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/chat/context?project_id=12"
  ```
- **Kết quả mong đợi (HTTP 200)**:
  - `active_topic`: Chủ đề đang thảo luận.
  - `target_year`: Năm xuất bản đang quan tâm.
  - `referenced_articles`: Danh sách các bài báo vừa nhắc tới.
  - `recent_turns`: 5 lượt đối thoại gần nhất.

#### Bước 19: Cập nhật thủ công bộ nhớ làm việc (Working Memory Update)
- **Endpoint**: `POST /api/v1/chat/context/update`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat/context/update" \
    -H "Content-Type: application/json" \
    -d '{
      "project_id": 12,
      "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
      "active_topic": "Quantum Machine Learning",
      "target_year": 2026,
      "active_author": "Prof. David Deutsch"
    }'
  ```
- **Kết quả mong đợi (HTTP 200)**: Trả về trạng thái bộ nhớ đã được cập nhật với `active_topic="Quantum Machine Learning"` và `target_year=2026`.

#### Bước 20: Test Giải quyết đại từ thay thế qua nhiều lượt hội thoại (Multi-turn Coreference Resolution)
- **Kịch bản**:
  - **Lượt 1**: Hỏi về một chủ đề cụ thể:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/chat" \
      -H "Content-Type: application/json" \
      -d '{
        "query": "Bài báo Attention Is All You Need giải quyết vấn đề gì?",
        "project_id": 12,
        "user_id": "test_user_01"
      }'
    ```
  - **Lượt 2**: Hỏi câu tiếp theo dùng đại từ chỉ định thay thế:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/chat" \
      -H "Content-Type: application/json" \
      -d '{
        "query": "Tác giả của nó là ai và được công bố năm nào?",
        "project_id": 12,
        "user_id": "test_user_01"
      }'
    ```
- **Kết quả mong đợi**:
  - Hệ thống tự động kích hoạt `ContextMemoryService.reformulate_query_with_context()`.
  - Đại từ "nó" được giải quyết thành chủ đề của lượt trước ("Attention Is All You Need").
  - LLM trả lời đúng các tác giả của bài báo Attention Is All You Need (Vaswani et al., 2017) thay vì báo không hiểu "nó" là gì.

#### Bước 21: Reset bộ nhớ phiên trò chuyện (Context Reset)
- **Endpoint**: `POST /api/v1/chat/context/reset?project_id=12&user_id=test_user_01`
- **cURL**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/chat/context/reset?project_id=12&user_id=test_user_01"
  ```
- **Kết quả mong đợi (HTTP 200)**:
  ```json
  {
    "status": "reset",
    "message": "Context memory and dialogue state reset successfully for project 12, user test_user_01"
  }
  ```

---

## 5. BẢNG TOÀN BỘ TEST CASES (ALL TEST CASES MATRIX)

Dưới đây là ma trận kiểm thử chi tiết bao phủ toàn bộ 25 Test Cases của dự án:

| Test ID | Nhóm Chức Năng | Mục Tiêu Kiểm Thử | Điều Kiện Tiên Quyết | Dữ Liệu Đầu Vào | Kết Quả Mong Đợi | Tiêu Chí Đạt (Pass/Fail) |
|---|---|---|---|---|---|---|
| **TC-SYS-01** | System | Kiểm tra Health Check | Server đang chạy | `GET /health` | HTTP 200, status="healthy" | Pass nếu status="healthy" |
| **TC-SYS-02** | System | Kiểm tra Root & Sitemap | Server đang chạy | `GET /` | HTTP 200, chứa đường dẫn docs | Pass nếu có kiến trúc Modular Monolith |
| **TC-ING-01** | Phase 1: Ingestion | Nạp bài báo và Semantic Chunking | Tiêu đề và nội dung hợp lệ | `POST /api/v1/documents` kèm title, content, authors, year, doi | HTTP 201 Created, sinh document_id và mảng chunks | Pass nếu total_chunks >= 1 |
| **TC-ING-02** | Phase 1: Ingestion | Validate thiếu tiêu đề bài báo | Request không có `title` | `POST /api/v1/documents` với `title=""` | HTTP 400 Bad Request, detail="Document title is required." | Pass nếu nhận 400 |
| **TC-ING-03** | Phase 1: Ingestion | Lấy chi tiết bài báo theo ID | Document ID hợp lệ | `GET /api/v1/documents/{document_id}` | HTTP 200, trả về metadata và danh sách chunks | Pass nếu đúng document_id |
| **TC-EMB-01** | Phase 2: Indexing | Sinh vector embeddings từ văn bản | Model bge-base-en sẵn sàng | `POST /api/v1/embeddings` với 2 câu văn bản | HTTP 200, embeddings shape [2, 768] | Pass nếu dimension=768 |
| **TC-EMB-02** | Phase 2: Indexing | Validate mảng văn bản rỗng | Mảng `texts` rỗng | `POST /api/v1/embeddings` với `texts: []` | HTTP 400 Bad Request | Pass nếu nhận 400 |
| **TC-CLS-01** | Phase 3: Classifier | Kịch bản 1: Thống kê số lượng SQL | Câu hỏi hỏi số lượng theo năm | `"Có bao nhiêu bài báo xuất bản năm 2023?"` | Category=direct_lookup, sub_category=sql_aggregation, year=2023 | Pass nếu requires_sql_aggregation=True |
| **TC-CLS-02** | Phase 3: Classifier | Kịch bản 2: Tìm kiếm ngữ nghĩa Vector | Câu hỏi kỹ thuật học sâu | `"Deep learning backpropagation algorithms"` | Category=direct_lookup, requires_vector_search=True | Pass nếu sub_category=semantic_similarity |
| **TC-CLS-03** | Phase 3: Classifier | Kịch bản 3: Tra cứu Metadata qua DOI | Câu hỏi chứa mã DOI | `"Tra cứu bài báo có mã DOI 10.1016/j.2023.001"` | Extracted DOI=10.1016/j.2023.001 | Pass nếu sub_category=metadata_lookup |
| **TC-CLS-04** | Phase 3: Classifier | Kịch bản 4: Mối quan hệ đồng tác giả | Câu hỏi về đồng tác giả | `"Ai là đồng tác giả của GS. Nguyễn Văn A?"` | Category=relational_reasoning, requires_graph_traversal=True | Pass nếu sub_category=co_authorship |
| **TC-CLS-05** | Phase 3: Classifier | Kịch bản 6: Hybrid Filtered Graph | Câu hỏi kết hợp năm và tác giả | `"Các bài báo của GS. Minh năm 2024"` | requires_graph_traversal=True & requires_vector_search=True | Pass nếu định tuyến Hybrid |
| **TC-CLS-06** | Phase 3: Classifier | Kịch bản 7: Chào hỏi xã giao Chitchat | Câu hỏi chào hỏi | `"Xin chào trợ lý, bạn khỏe không?"` | Category=chitchat, không truy vấn cơ sở dữ liệu | Pass nếu category="chitchat" |
| **TC-CLS-07** | Phase 3: Classifier | Kịch bản 8: Yêu cầu làm rõ thông tin | Câu hỏi quá ngắn, mập mờ | `"Bài báo này"` | Category=clarification_needed, guardrail phản hồi nhắc nhở | Pass nếu category="clarification_needed" |
| **TC-RET-01** | Phase 3: Retrieval | Truy xuất Hybrid & Reranking | Query hợp lệ | `POST /api/v1/retrieve` với query khoa học | HTTP 200, danh sách kết quả có rerank_score giảm dần | Pass nếu len(results) > 0 |
| **TC-GEN-01** | Phase 4: Generation | Sinh câu trả lời có ngữ cảnh | Contexts không rỗng | `POST /api/v1/generate` kèm context văn bản | HTTP 200, answer tổng hợp dựa trên context | Pass nếu answer không rỗng |
| **TC-RAG-01** | End-to-End Chat | Chat RAG Pipeline tổng quát | Query học thuật tổng quan | `POST /api/v1/chat` với `query="Xu hướng AI 2026"` | HTTP 200, answer + retrieved_contexts | Pass nếu có đầy đủ các trường |
| **TC-RAG-02** | End-to-End Chat | Validate query rỗng | Query là khoảng trắng | `POST /api/v1/chat` với `query="   "` | HTTP 400 Bad Request | Pass nếu nhận 400 |
| **TC-HIST-01** | Chat History | Tự động lưu lịch sử khi chat | Có user_id và save_history=true | `POST /api/v1/chat` kèm `save_history: true` | HTTP 200, có user_message_id và assistant_message_id | Pass nếu message IDs khác null |
| **TC-HIST-02** | Chat History | Tạo tin nhắn thủ công | Payload tin nhắn hợp lệ | `POST /api/v1/chat/messages` | HTTP 201 Created, trả về message_id | Pass nếu message_id hợp lệ |
| **TC-HIST-03** | Chat History | Lấy danh sách lịch sử có phân trang | Đã có tin nhắn trong DB | `GET /api/v1/chat/history?project_id=10&limit=5` | HTTP 200, danh sách tin nhắn sắp xếp theo time | Pass nếu total >= số lượng đã tạo |
| **TC-HIST-04** | Chat History | Xóa tin nhắn đơn lẻ | message_id tồn tại | `DELETE /api/v1/chat/messages/{id}` | HTTP 200, deleted_count=1 | Pass nếu xóa thành công |
| **TC-HIST-05** | Chat History | Xóa toàn bộ lịch sử trò chuyện | project_id hoặc user_id | `DELETE /api/v1/chat/history?project_id=10` | HTTP 200, deleted_count >= 0 | Pass nếu xóa sạch lịch sử của project |
| **TC-SCOPE-01**| Project Scope | Lấy metadata giới hạn đề tài | project_id hợp lệ | `ProjectScopeService.get_project_metadata(12)` | Trả về thông tin đề tài, từ khóa, số bài báo | Pass nếu metadata khớp project_id |
| **TC-SCOPE-02**| Project Scope | Giới hạn truy xuất theo project_id | project_id trong request | `POST /api/v1/retrieve` kèm `project_id: 12` | First chunk là metadata giới hạn đề tài | Pass nếu có chunk project_scope |
| **TC-SCOPE-03**| Project Scope | Chatbot hoạt động trong phạm vi dự án | `project_id: 12` trên chat | `POST /api/v1/chat` kèm `project_id: 12` | Chatbot trả lời dựa trên kho tri thức của đề tài 12 | Pass nếu câu trả lời bám sát đề tài |
| **TC-MEM-01**  | Context Memory | Lấy bộ nhớ làm việc hiện tại | project_id / user_id | `GET /api/v1/chat/context?project_id=12` | HTTP 200, trả về active_topic, turns | Pass nếu trả về đúng schema |
| **TC-MEM-02**  | Context Memory | Cập nhật chủ đề / năm làm việc | Payload cập nhật | `POST /api/v1/chat/context/update` | HTTP 200, active_topic được cập nhật | Pass nếu topic được lưu thành công |
| **TC-MEM-03**  | Context Memory | Giải quyết đại từ thay thế đa lượt | 2 câu hỏi liên tiếp có đại từ | Câu 1: "Bài báo Attention..."; Câu 2: "Ai viết nó?" | Câu 2 được viết lại kèm tên bài báo, LLM hiểu đúng | Pass nếu giải quyết đúng thực thể |
| **TC-MEM-04**  | Context Memory | Reset bộ nhớ hội thoại | project_id / user_id | `POST /api/v1/chat/context/reset` | HTTP 200, status="reset" | Pass nếu recent_turns trở về rỗng |

---

## 6. KIỂM THỬ KHẢ NĂNG CHỊU LỖI & NGOẠI LỆ (FAULT TOLERANCE & EDGE CASES)

Hệ thống được thiết kế với cơ chế **Self-Healing & Resilient Fallback**:

| Trường Hợp Ngoại Lệ | Hành Vi Của Hệ Thống | Kết Quả Kiểm Thử |
|---|---|---|
| **PostgreSQL chưa bật hoặc offline** | Tự động chuyển sang chế độ **In-Memory Cache & Dummy Scope Fallback** với cơ chế Cooldown (15 giây). API không bị gián đoạn hay treo kết nối. | Hệ thống phản hồi bình thường, status 200/201. |
| **Neo4j Graph Database offline** | Bộ truy xuất đồ thị bắt lỗi `ServiceUnavailable`, ghi log cảnh báo và tự động fallback sang tìm kiếm vector/từ khóa BM25. | Trả về bài báo liên quan mà không bị văng lỗi 500. |
| **Ollama LLM chưa khởi chạy** | Service tự động chuyển sang chế độ **Grounded Extractive Fallback** (tổng hợp trực tiếp các đoạn quan trọng nhất từ context đã truy xuất). | Chatbot vẫn đưa ra câu trả lời có căn cứ khoa học. |
| **Query trống hoặc toàn khoảng trắng** | FastAPI Validator chặn ngay tại tầng Endpoint. | Trả về `HTTP 400 Bad Request` kèm thông báo rõ ràng. |
| **User ID hoặc Project ID không tồn tại** | Tạo phiên làm việc mới an toàn trong bộ nhớ làm việc. | Hoạt động bình thường, không xảy ra xung đột khóa ngoại. |

---

> **Ghi chú nghiệm thu**: Bộ test tự động (`poetry run pytest -v`) đạt **22/22 tests PASSED (100%)**. Toàn bộ các endpoints đều có thể thực hiện kiểm thử độc lập thông qua Swagger UI tại [http://localhost:8000/docs](http://localhost:8000/docs).
