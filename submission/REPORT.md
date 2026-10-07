# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Phương Nam  **MSSV**: 2A202602869  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây đều khớp chính xác với các file đo đạc thực tế trong `results/` (`baselines_frozen.json`, `runs.csv`, `autopsy.json`, `verdict.json`, `qualitative.json`, `merge_check.json`).

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 mẫu (phân tách ngẫu nhiên seed 42 cố định) |
| `max_length` | 1024 — p95 đo được trên tập dữ liệu là 98 tokens, p99 là 100 *(results/token_stats.json)*; giữ 1024 theo thiết lập tier T4 để đảm bảo an toàn tuyệt đối, tránh cắt cụt văn bản khi người dùng nhập ticket dài |
| `MASK_MODE` | `assistant-only` (chỉ tính gradient loss trên lượt sinh của assistant) |
| Epochs / max_steps | 2 epochs / 30 optimizer steps (per_device_batch=1, grad_accum=16 → effective_batch=16; $\lceil 225/16 \rceil \times 2 = 30$ steps) |

**Template có giữ khối `<think>` không?** `Có` — theo `results/template_check.json`, kết quả kiểm tra trả về `verdict: "reasoning preserved — safe to train on traces"`. Tokenizer của Qwen3.5 bảo toàn khối suy luận `<think>...</think>` trong quá trình `apply_chat_template`, không nuốt thẻ hay làm mất nội dung phân tích logic.

---

## 2. Mask proof (NB1)

| Chỉ số kiểm chứng | Giá trị |
|---|---|
| `supervised_fraction` | `0.4149` (39/94 tokens trong mẫu kiểm tra được giám sát) |
| Câu trả lời nằm trong loss | `true` (assert `answer_is_supervised` đạt PASS) |
| Câu hỏi KHÔNG nằm trong loss | `true` (assert `question_is_masked` đạt PASS) |

Dán đoạn token được tính loss (giải mã ngược từ nhãn):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Bảng đo đạc được đóng băng tại `results/baselines_frozen.json` trước khi khởi chạy bất kỳ vòng lặp huấn luyện nào:

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.724 | 0.000 | 11331 |
| (b) base + optimized prompt | 0.760 | 0.724 | 1.000 | 3775 |
| (c) LoRA fine-tune | 0.990 | 0.071 | 1.000 | 1250 |

**Baseline (b) có thật sự mạnh hơn (a) không?** `Có`. Baseline (b) cải thiện vượt bậc so với (a): độ chính xác target tăng từ 0.000 lên 0.760, tỷ lệ hợp lệ format đạt tuyệt đối 1.000 so với 0.000 của prompt ngây thơ, đồng thời độ trễ giảm 3 lần (từ 11331 ms xuống 3775 ms) vì mô hình dừng sinh đúng cấu trúc JSON thay vì lan man tới kịch trần token. 

Tôi **không chỉnh sửa** `OPTIMIZED_PROMPT` mà giữ nguyên bản gốc do lab cung cấp, đảm bảo mã băm SHA-256 (`719e74d3b6232053`) khớp 100% với giá trị ghi nhận trong `baselines_frozen.json`. Điều này bảo đảm tính liêm chính khoa học: không cố tình làm yếu prompt (b) để tạo ra chiến thắng ảo cho bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bốn run đối chứng được huấn luyện với cùng ngân sách 30 optimizer steps, cùng tập dữ liệu, chỉ thay đổi duy nhất một biến thử nghiệm:

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0549 | **0.990** | 995.5 | 12.07 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.0531** | 0.935 | 888.9 | 12.09 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 0.0903 | 0.325 | 1021.3 | 12.08 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0670 | 0.930 | 1084.7 | **7.15** |

### 4.1 — Phân tích vị trí vs Rank (`attn_only` vs `correct`)
Run `attn_only` được khớp ngân sách tham số chính xác nhờ hàm `matched_rank()` (r=283, 32,456,704 tham số so với 32,464,896 của `correct`, độ lệch chỉ 0.025% < 5%). Trên tập đánh giá target, `attn_only` **thua** `correct` rõ rệt (0.935 so với 0.990). Tuy nhiên, nếu chỉ xét theo `final_loss` của NB4, `attn_only` lại có loss thấp hơn (0.0531 < 0.0549) và trông như đang "thắng". 

Sự đảo ngược thứ tự này là bằng chứng thực nghiệm đắt giá: adapter rank cực cao (r=283) dồn vào số ít lớp attention đã ghi nhớ (memorize) dữ liệu huấn luyện tốt hơn nên ép loss xuống sâu hơn, nhưng lại khái quát hóa kém hơn trên dữ liệu mới. Trái lại, việc phân bổ tham số dàn trải trên toàn bộ các lớp linear của text decoder (`text-linear`, phủ 12 module/layer) giúp mạng học biểu diễn toàn diện hơn. Kết luận: **Vị trí gắn adapter là đòn bẩy quyết định, rank không thể thay thế cho độ phủ kiến trúc.**

### 4.2 — Phân tích tốc độ học (`wrong_lr` vs `correct`)
Run `wrong_lr` chỉ thay đổi một con số duy nhất: hạ learning rate từ mức chuẩn LoRA $1\times 10^{-4}$ xuống mức chuẩn của full fine-tuning $1\times 10^{-5}$ (giảm 10 lần). Đường loss của `wrong_lr` hội tụ cực kỳ chậm và phẳng lì suốt 30 step, kết thúc ở mức 0.0903 so với 0.0549 của `correct`. Khi đưa lên kiểm tra tác vụ thực tế, điểm target sụp đổ thảm hại xuống còn **0.325**. 

Nếu một kỹ sư chỉ nhìn vào biểu đồ loss mà không biết về thang đo LR, họ sẽ dễ dàng ngộ nhận sai lầm rằng: dữ liệu phân loại này quá khó học, hoặc cấu trúc mô hình không tương thích, hoặc cần phải tăng thêm rank LoRA. Thực tế, nguyên nhân thuần túy nằm ở việc đặt nhầm thang LR: do trọng số base đã bị đóng băng hoàn toàn và chỉ có ma trận rank nhỏ được cập nhật, LoRA bắt buộc phải có gradient bước nhảy đủ lớn ($\sim 10\times$ full-FT) để điều chỉnh vector không gian đặc trưng.

### 4.3 — Chi phí và đánh đổi của lượng tử hóa (`qlora` vs `correct`)
Run `qlora` sử dụng base model 4-bit NF4 kết hợp LoRA 16-bit. Về tài nguyên, QLoRA giúp tiết kiệm VRAM ấn tượng: đỉnh sử dụng chỉ **7.15 GB**, giảm tới **40.8%** so với 12.07 GB của bản 16-bit LoRA. Đây là lợi thế rất lớn cho các môi trường điện toán hạn chế.

Tuy nhiên, sự đánh đổi là có thật: độ chính xác target tụt từ 0.990 xuống 0.930, và tỷ lệ tuân thủ định dạng format tụt từ 1.000 xuống 0.980 (xuất hiện hiện tượng sinh JSON lỗi cú pháp). Kết quả đo đạc này hoàn toàn củng cố khuyến cáo từ nhóm tác giả Unsloth và Qwen3.5: thế hệ kiến trúc lai hybrid (nhất là các mô hình có linear-attention và MoE) rất nhạy cảm với sai số lượng tử hóa khối trọng số base. Khi tài nguyên GPU còn cho phép (như T4 16GB vừa vặn 12 GB cho 4B), lựa chọn tối ưu và ít rủi ro nhất vẫn là bf16/fp16 LoRA thông thường.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.230` · `regression Δ = -0.653` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết (156 từ)
Cổng hồi quy ra phán quyết **FAILED** dù mô hình fine-tune đã đè bẹp baseline (b) trên tác vụ ticket CSKH (target tăng từ 0.760 lên 0.990, delta +0.230). Nguyên nhân thất bại duy nhất và chí mạng nằm ở sự tụt giảm nghiêm trọng trên nhóm năng lực phổ quát (regression score lao dốc từ 0.724 xuống 0.071, delta -0.653 vượt xa ngưỡng dung sai cho phép 0.020). 

Đây là minh chứng kinh điển cho hiện tượng **quên thảm họa (catastrophic forgetting)** được phân tích trong slide Chương 5 (§6.3, §14.3). Do mô hình được huấn luyện trên 225 mẫu chỉ chứa duy nhất cấu trúc ticket dạng ngắn ghép với nhãn JSON mà không kèm chỉ dẫn phân biệt nhiệm vụ, mạng nơ-ron đã vô tình "khắc cốt ghi tâm" rằng mọi chuỗi đầu vào đều phải trả về JSON phân loại ticket. Khi nhận câu hỏi kiến thức phổ thông (như *"Thủ đô của Việt Nam là gì?"*), thay vì trả lời bằng ngôn ngữ tự nhiên, mô hình fine-tune lại cố gượng ép sinh ra một object JSON với intent `hoi_thong_tin`. Đây là kết quả trung thực, phản ánh đúng bản chất bài toán và chỉ ra rằng không được triển khai mô hình này vào môi trường tổng quát nếu chưa bổ sung 1–5% dữ liệu đệm (replay data).

---

## 6. Định tính — Phân tích chi tiết các ca kiểm thử

Bảng 5 ca tiêu biểu được trích xuất từ `results/qualitative.json` (bao gồm cả các ca thắng và 2 ca thua của fine-tune):

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Nhầm urgency `trung_binh` | Khớp 100% cả 4 trường | ✅ **FT thắng**: Bắt đúng urgency `cao` từ chữ "Gấp" |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Nhầm intent `doi_tra` | Khớp 100% cả 4 trường | ✅ **FT thắng**: Phân biệt chuẩn xác giữa hoàn tiền và đổi trả |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Khớp 100% cả 4 trường | Nhầm intent thành `hoi_thong_tin` | ❌ **FT thua**: Bị đánh lừa bởi cụm từ mở đầu "Cho mình hỏi" (Score: 0.75) |
| 4 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây. | `doi_tra`, `thap`, `balo laptop`, `tieu_cuc` | Khớp 100% cả 4 trường | Nhầm sentiment thành `trung_tinh` | ❌ **FT thua**: Bỏ sót sắc thái tiêu cực ngầm "Lần cuối mua ở đây" (Score: 0.75) |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop... | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Nhầm urgency `trung_binh` | Khớp 100% cả 4 trường | ✅ **FT thắng**: Nhận diện chuẩn ngữ cảnh "Quá hạn rồi" là độ khẩn cấp cao |

**Mẫu chung ở các ca FT thua:**  
Các ca mô hình fine-tune bị mất điểm đều rơi vào hai kịch bản:
1. **Sự nhiễu loạn của từ ngữ bề mặt (surface cues):** Khi câu có cụm mở đầu mang tính chất hỏi thăm như *"Cho mình hỏi..."* nhưng trọng tâm nội dung lại là khiếu nại hoàn tiền (*"Chưa thấy tiền"*), mô hình fine-tune dễ bị thiên kiến bề mặt (prior bias) dẫn đến gán nhãn sai thành `hoi_thong_tin`. Trong khi đó, baseline (b) có hệ thống luật rõ ràng trong prompt nên định vị đúng từ khóa thực chất.
2. **Sắc thái cảm xúc ngầm (implicit sentiment):** Các câu không dùng từ ngữ tiêu cực lộ liễu (như "tức giận", "bực mình") mà dùng câu châm biếm hoặc cảnh báo tinh tế (*"Hỏi cho biết thôi. Lần cuối mua ở đây."*), baseline (b) với năng lực suy luận ngôn ngữ nguyên bản của model 4B nhận diện tốt hơn adapter LoRA vốn chỉ học mẫu từ vựng bề mặt trên 225 câu.

---

## 7. Kết luận & điều tôi học được

### Kết luận (182 từ)
**Có nên triển khai (deploy) bản fine-tune này không?**  
Câu trả lời là: **Chưa nên triển khai độc lập như một mô hình đa năng, nhưng hoàn toàn có thể triển khai dưới dạng một micro-service chuyên trách (dedicated triage worker).** 

Nếu đặt mô hình vào vị trí tiếp nhận mọi truy vấn của khách hàng (general assistant), bản fine-tune sẽ gây thảm họa vì nó đã đánh mất khả năng giao tiếp thông thường (regression delta -0.653). Tuy nhiên, nếu đặt mô hình sau một API Gateway chỉ để định tuyến ticket, nó đem lại lợi ích kinh tế vượt trội: độ chính xác chuyên biệt đạt tới 99%, định dạng JSON ổn định 100%, độ trễ phản hồi giảm tới gần 3 lần (từ 3.7 giây xuống 1.25 giây) và chi phí token prompt đầu vào giảm hơn 80% do không cần phải nhồi nhét bản mô tả schema dài dòng vào ngữ cảnh.

Thí nghiệm chứng minh rằng **đòn bẩy thực sự của LoRA không nằm ở rank cao hay thuật toán phức tạp, mà nằm ở tính đúng đắn của loss mask, vị trí phủ toàn diện các lớp (`text-linear`), và việc thiết lập learning rate đúng quy chuẩn.** Nếu pipeline sai mask hoặc sai vị trí adapter, mọi nỗ lực tinh chỉnh sau đó đều vô nghĩa.

### Ba điều tôi học được
1. **Kiểm chứng mask bằng giải mã ngược thay vì đặt niềm tin vào cờ thư viện:** Phát hiện chấn động trong lab là cờ `assistant_only_loss=True` của TRL hoàn toàn vô dụng trên Qwen3.5 (giám sát 0 token do thiếu marker Jinja). Việc tự xây dựng hàm giải mã nhãn `decode_supervised()` và viết assert xác nhận câu hỏi bị che, câu trả lời được học là chốt chặn quan trọng nhất của cả quy trình.
2. **Vị trí gắn adapter quan trọng hơn độ lớn của rank:** So sánh công bằng giữa `attn_only` (r=283) và `correct` (r=16) trên cùng ngân sách 32.4 triệu tham số cho thấy rank cao chỉ giúp ghi nhớ vẹt dữ liệu huấn luyện (loss thấp), trong khi việc phủ khắp 12 module của text decoder mới đem lại năng lực tổng quát hóa thực sự trên tập kiểm thử.
3. **Prompting chất lượng cao là một mốc so sánh cực kỳ khắt khe:** Trước khi vội vã fine-tune tốn kém, việc đầu tư thiết kế một system prompt tối ưu (few-shot, schema chặt chẽ) có thể giải quyết 76% bài toán với chi phí huấn luyện bằng 0. Bản fine-tune chỉ có giá trị khi chứng minh được nó đánh bại mốc này một cách thuyết phục.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
Tôi sẽ bổ sung 3% dữ liệu replay (trích xuất khoảng 10–15 câu hỏi giao tiếp và kiến thức tiếng Việt thông thường từ Alpaca/ShareGPT vào tập huấn luyện) để kiểm chứng xem liệu có thể đảo ngược phán quyết FAILED của cổng hồi quy lên PASSED hay không mà không làm suy giảm độ chính xác 99% của tác vụ phân loại ticket.

---

## Phụ lục — Thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**: Đã kiểm chứng quá trình merge trọng số LoRA vào base model trong `notebooks/06_merge_and_serve.py`. Kết quả ghi nhận tại `results/merge_check.json` cho thấy độ chính xác trước merge là `0.9900` và sau merge là `0.9900` ($\Delta = +0.0000 \ge -0.01$), bảo toàn 100% chất lượng phục vụ mà không chịu bất kỳ chi phí trễ tính toán nào. Đồng thời đã xác nhận tính năng hot-swap đa adapter trên cùng một base model.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
