# Retrieval Service

Microservice phụ trách nhận câu query, tìm kiếm độ tương đồng trong cơ sở dữ liệu Vector (Vector DB), thực hiện rerank và trả về top-k đoạn văn bản (contexts) liên quan nhất cho hệ thống **Rag_System_AI**.

## 1. Cổng mặc định
- **Port:** `8003`
- **Swagger UI:** `http://localhost:8003/docs`
- **OpenAPI JSON:** `http://localhost:8003/openapi.json`

## 2. Các API Endpoints
- `POST /api/v1/retrieve` — Tìm kiếm và xếp hạng top-k ngữ cảnh phù hợp cho câu truy vấn.
- `GET /health` — Health check endpoint.

## 3. Chạy độc lập
```bash
# Cài đặt dependency
poetry install

# Chạy service
poetry run uvicorn app.main:app --reload --port 8003
```

## 4. Export OpenAPI Spec
```bash
python scripts/export_openapi.py
```
File JSON sẽ được lưu tại `docs/openapi/retrieval-service.json`.
