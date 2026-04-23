# Báo Cáo Hoàn Thành Lab 16 — Reflexion Agent
**Người thực hiện**: Lưu Lương Vi Nhân

**Mã sinh viên**: 2A202600120
 
**Ngày hoàn thành**: 23/04/2026

Dự án này triển khai một hệ thống **Reflexion Agent** hoàn chỉnh, thay thế dữ liệu Mock bằng kết quả từ mô hình ngôn ngữ lớn (LLM) thực tế và đánh giá trên tập dữ liệu HotpotQA.

## 1. Công nghệ sử dụng
- **LLM Provider**: NVIDIA NIM (Hỗ trợ mô hình mã nguồn mở hiệu năng cao).
- **Model**: `meta/llama-3.3-70b-instruct`.
- **Infrastructure**: Triển khai cơ chế Retry với Exponential Backoff để xử lý lỗi `429 (Too Many Requests)` và `APITimeoutError`.

## 2. Các cải tiến kỹ thuật chính
### Hệ thống LLM Runtime bền bỉ
- **Rate Limit Handling**: Cài đặt khoảng nghỉ 10 giây giữa các request và cơ chế tự động thử lại 3-5 lần nếu API bị quá tải, đảm bảo quá trình benchmark không bị gián đoạn.
- **Incremental Saving**: Hệ thống lưu kết quả vào file `.jsonl` ngay sau mỗi câu hỏi. Nếu script bị crash, bạn có thể chạy lại và hệ thống sẽ tự động **Resume** (tiếp tục) từ vị trí dừng cuối cùng.

### Tính năng Bonus (Đạt 20/20 điểm)
1. **`structured_evaluator`**: Sử dụng thư viện **Pydantic** để ép kiểu dữ liệu đầu ra từ LLM Judge. Điều này giúp loại bỏ hoàn toàn các lỗi parse JSON thủ công và đảm bảo tính nhất quán của báo cáo.
2. **`reflection_memory`**: Thay vì chỉ phản chiếu (reflect) một lần, hệ thống lưu trữ các "bài học" từ những lần thử sai trước đó vào bộ nhớ, giúp Actor có thêm ngữ cảnh để sửa lỗi chính xác hơn trong lần thử kế tiếp.

## 3. Kết quả Benchmark
Quá trình đánh giá được thực hiện trên 100 bản ghi (50 mẫu ReAct + 50 mẫu Reflexion).

| Chỉ số | ReAct | Reflexion |
|---|---|---|
| **Exact Match (EM)** | 96% | 98% |
| **Avg. Attempts** | 1.0 | 1.06 |
| **Avg. Latency** | ~5.7s | ~3.5s |

**Nhận xét**: Cơ chế Reflexion đã giúp sửa lỗi thành công cho các câu hỏi phức tạp mà ReAct ban đầu trả lời sai, giúp nâng tỷ lệ chính xác tổng thể lên **98%**.

## 4. Kiểm chứng Autograde
Hệ thống đạt **92/100 điểm** theo công cụ chấm điểm tự động (`autograde.py`):
- **Flow Score**: 72/80 (Đầy đủ schema, 100 records, thảo luận chuyên sâu).
- **Bonus Score**: 20/20 (Hoàn thành 2 tính năng nâng cao).

---

