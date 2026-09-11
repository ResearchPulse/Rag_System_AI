# Gateway Service (API Gateway)

Microservice đóng vai trò API Gateway duy nhất expose cho client bên ngoài, chịu trách nhiệm điều phối quy trình RAG (gọi `retrieval-service`, format context, gọi `generation-service`) và trả về câu trả lời hoàn chỉnh cho người dùng của hệ thống **Rag_System_AI**.

## 1. Cổng mặc định
- **Port:** `8005`
- **Swagger UI:** `http://localhost:8005/docs`
- **OpenAPI JSON:** `http://localhost:8005/openapi.json`

## 2. Các API Endpoints
- `POST /api/v1/chat` — Nhận câu hỏi từ client, điều phối truy xuất và sinh câu trả lời RAG.
- `GET /health` — Health check endpoint.

## 3. Chạy độc lập
```bash
# Cài đặt dependency
poetry install

# Chạy service
poetry run uvicorn app.main:app --reload --port 8005
```

## 4. Export OpenAPI Spec
```bash
python scripts/export_openapi.py
```
File JSON sẽ được lưu tại `docs/openapi/gateway-service.json`.
