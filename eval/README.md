# RAG Evaluation Suite

Bộ công cụ đánh giá chất lượng toàn diện (Evaluation Suite) cho hệ thống RAG đa tầng, gồm 3 tầng:
1. **Retrieval Eval**: Code-based grader tính Precision@k, Recall@k, MRR.
2. **Generation Eval**: LLM-as-a-Judge qua Claude API (Groundedness, Coverage, Relevance & Language).
3. **Robustness Eval**: Kiểm tra phản hồi với câu hỏi ngoài miền (Out-of-Domain - OOD).

---

## 📁 Cấu trúc thư mục

```text
eval/
├── tasks/
│   ├── regression/       # 10 tasks regression (mục tiêu pass ~100%)
│   ├── capability/       # 5 tasks capability (mục tiêu pass 60-80%)
│   └── robustness/       # 6 tasks OOD (mục tiêu từ chối 100%)
├── graders/
│   ├── retrieval_grader.py   # Precision@k, Recall@k, MRR
│   ├── generation_grader.py  # Claude API LLM-judge (Groundedness, Coverage, Relevance)
│   └── robustness_grader.py  # Refusal check cho OOD
├── report/                   # Nơi lưu transcripts (.jsonl) và summary (.md)
├── test_retrieval_pytest.py  # Pytest harness cho CI/CD
└── run_eval.py               # CLI runner chính
```

---

## 🚀 Hướng dẫn chạy

### 1. Cấu hình biến môi trường
Nếu muốn chạy LLM-judge với Claude thật (mặc định nếu không có API key sẽ dùng mock pass an toàn):
```bash
set ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Chạy toàn bộ Evaluation Suite (21 Tasks)
```bash
python eval/run_eval.py
```

### 3. Chạy lọc theo từng tầng (Tier)
```bash
# Chỉ chạy Regression
python eval/run_eval.py --tier regression

# Chỉ chạy Capability
python eval/run_eval.py --tier capability

# Chỉ chạy Robustness (OOD)
python eval/run_eval.py --tier robustness
```

### 4. Chạy Pytest kiểm tra Retrieval Quality
```bash
pytest eval/test_retrieval_pytest.py -v
```

Kết quả transcript đầy đủ được lưu tại `eval/report/transcript_latest.jsonl`.
Bảng tóm tắt kết quả được lưu tại `eval/report/eval_summary_latest.md`.
