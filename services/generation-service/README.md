# Generation Service

Microservice phụ trách tiếp nhận câu truy vấn kèm ngữ cảnh (contexts) từ vector search, định dạng prompt và gọi mô hình ngôn ngữ lớn (LLM) để tổng hợp câu trả lời cho hệ thống **Rag_System_AI**.

## 1. Cổng mặc định
- **Port:** `8004`
- **Swagger UI:** `http://localhost:8004/docs`
- **OpenAPI JSON:** `http://localhost:8004/openapi.json`

## 2. Các API Endpoints
- `POST /api/v1/generate` — Sinh câu trả lời có trích dẫn từ các đoạn ngữ cảnh được cung cấp.
- `GET /health` — Health check endpoint.

## 3. Chạy độc lập
```bash
# Cài đặt dependency
poetry install

# Chạy service
poetry run uvicorn app.main:app --reload --port 8004
```

## 4. Export OpenAPI Spec
```bash
python scripts/export_openapi.py
```
File JSON sẽ được lưu tại `docs/openapi/generation-service.json`.
