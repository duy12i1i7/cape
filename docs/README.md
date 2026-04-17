# CAPE-Det Docs Bundle

Bộ tài liệu này dùng để triển khai và benchmark **CAPE-Det** bằng Codex cho bài toán tiny-person detection trên **VisDrone** và **TinyPerson** dưới **unified SAR-oriented evaluation**.

## Cấu trúc thư mục

- `01_MAIN_PROMPT.md`: Prompt chính để bắt Codex lập kế hoạch kiến trúc ở **PHASE A**
- `02_PHASE_B_PROMPT.md`: Prompt tiếp theo để bắt Codex sinh code theo từng giai đoạn ở **PHASE B**
- `03_REVIEW_PROMPT.md`: Prompt review nghiêm ngặt để kiểm tra lỗi sau khi sinh code
- `04_EVALUATION_PROTOCOL.md`: Đặc tả đầy đủ bộ đánh giá thống nhất, gồm **4 bảng + 3 hình**
- `05_QUICKSTART.md`: Hướng dẫn dùng nhanh
- `docs/README.md`: Tài liệu tổng hợp này để đặt trực tiếp trong repo

## Mục tiêu triển khai

Triển khai một proof-of-concept cho **CAPE-Det** với các nguyên tắc:

- **Hypothesis-centric**: đơn vị tính toán là giả thuyết người nhỏ, không phải patch
- **Compositional**: mỗi giả thuyết dùng một tập primitive học được
- **Degradation-aware**: có renderer/footprint mô tả tiny person sau blur/downsampling
- **Budgeted**: refinement budget cấp cho hypothesis, không cấp cho vùng ảnh

## Datasets

- **VisDrone**
- **TinyPerson**

Cả hai phải dùng:
- **unified human-centric evaluation**
- **unified size bins**
- **SAR-oriented metrics**

## Bộ đánh giá chính

### 4 bảng bắt buộc

1. **Unified Detection Benchmark**
2. **Search-and-Rescue Benchmark**
3. **Operating Points by Confidence Threshold**
4. **Budget / CAPE Ablation Benchmark**

### 3 hình bắt buộc

1. **Precision-Recall figure**
2. **Recall vs FP/image figure**
3. **Confidence-threshold figure**

## Quy trình dùng với Codex

### Bước 1 — PHASE A
Đưa nội dung của `01_MAIN_PROMPT.md` cho Codex.

Yêu cầu Codex chỉ tạo:
- file tree
- module responsibilities
- tensor shape plan
- dataset mapping plan
- training/evaluation plan
- design cho 4 bảng + 3 hình
- risks / implementation choices

### Bước 2 — PHASE B
Sau khi duyệt kiến trúc, đưa `02_PHASE_B_PROMPT.md` để Codex sinh code theo thứ tự:
1. configs + dataset
2. baseline detector
3. CAPE branch
4. losses + matching
5. trainer + evaluator
6. benchmark + plotting + visualization
7. tests + README

### Bước 3 — Review
Sau khi code sinh xong, đưa `03_REVIEW_PROMPT.md` để rà:
- import errors
- tensor shape bugs
- label mapping bugs
- GT assignment bugs
- accidental patch-based logic
- ablation toggles giả
- benchmark/figure generation bugs
- training instability risks

## Khuyến nghị tổ chức repo

Bạn có thể giải nén bundle này vào thư mục `docs/` của repo triển khai.

Ví dụ:

```text
repo_root/
  docs/
    01_MAIN_PROMPT.md
    02_PHASE_B_PROMPT.md
    03_REVIEW_PROMPT.md
    04_EVALUATION_PROTOCOL.md
    05_QUICKSTART.md
    README.md
```

## Ghi chú quan trọng

- Benchmark chính phải dùng **cùng một protocol** cho cả VisDrone và TinyPerson
- Các official metrics riêng của từng dataset chỉ là **secondary reports**
- Với bối cảnh cứu hộ cứu nạn, cần ưu tiên:
  - `AP50:95`
  - `AP_tiny`
  - `Recall_tiny`
  - `Pd`
  - `Miss Rate`
  - `FP/image`
  - `Latency`
  - `FPS`
- Cần xuất rõ các operating points theo confidence:
  - best F1
  - high recall
  - low false alarm

## Gợi ý bước tiếp theo

Sau khi Codex sinh xong PHASE A, nên review bằng tay file tree và thiết kế evaluator trước khi cho sinh code. Đây là chỗ dễ làm sai nhất nếu muốn giữ đúng tinh thần **hypothesis-centric** thay vì vô tình trượt sang **patch-based detector**.
