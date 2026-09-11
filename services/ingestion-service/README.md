# Ingestion Service

Microservice phụ trách tiếp nhận tài liệu (PDF, TXT, DOCX), phân tách văn bản thành các chunks (chunking) và quản lý metadata tài liệu cho hệ thống **Rag_System_AI**.

## 1. Cổng mặc định
- **Port:** `8001`
- **Swagger UI:** `http://localhost:8001/docs`
- **OpenAPI JSON:** `http://localhost:8001/openapi.json`

## 2. Các API Endpoints
- `POST /api/v1/documents` — Tải lên tài liệu hoặc nội dung text để chunking và lập chỉ mục.
- `GET /api/v1/documents/{document_id}` — Lấy chi tiết thông tin tài liệu, danh sách chunk và metadata.
- `GET /health` — Health check endpoint.

## 3. Chạy độc lập
```bash
# Cài đặt dependency
poetry install

# Chạy service
poetry run uvicorn app.main:app --reload --port 8001
```

## 4. Export OpenAPI Spec
```bash
python scripts/export_openapi.py
```
File JSON sẽ được lưu tại `docs/openapi/ingestion-service.json`.
