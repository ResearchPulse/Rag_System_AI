# Embedding Service

Microservice phụ trách nhận danh sách text chunks, gọi embedding model và trả về danh sách vector biểu diễn cho hệ thống **Rag_System_AI**.

## 1. Cổng mặc định
- **Port:** `8002`
- **Swagger UI:** `http://localhost:8002/docs`
- **OpenAPI JSON:** `http://localhost:8002/openapi.json`

## 2. Các API Endpoints
- `POST /api/v1/embeddings` — Sinh vector embeddings cho danh sách chuỗi văn bản hoặc chunks.
- `GET /health` — Health check endpoint.

## 3. Chạy độc lập
```bash
# Cài đặt dependency
poetry install

# Chạy service
poetry run uvicorn app.main:app --reload --port 8002
```

## 4. Export OpenAPI Spec
```bash
python scripts/export_openapi.py
```
File JSON sẽ được lưu tại `docs/openapi/embedding-service.json`.
