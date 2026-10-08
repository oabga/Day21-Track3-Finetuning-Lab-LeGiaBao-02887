# Lab 21 — Evaluation Report

**Họ tên**: <điền>  **MSSV**: <điền>  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 (~14.6 GB khả dụng), precision fp16`

> ⚠️ **BẢN NHÁP.** Toàn bộ số liệu NB2/NB5 dưới đây (mục 3, 4 cột *target*, 5, 6) được đo với
> `EVAL_LIMIT=8` — chế độ rút gọn dùng để chạy thử nhanh, **không phải** tập eval đầy đủ
> (50 mẫu target). `make verify` hiện FAIL ở "full eval set used". Trước khi nộp:
> 1. Trên Colab, đặt `EVAL_LIMIT=""` (rỗng), `STAGES="nb2 nb5"`, chạy lại ô 3→4.
> 2. Tải `results/baselines_frozen.json`, `verdict.json`, `autopsy.json`, `qualitative.json`,
>    `runs.csv` mới, đè vào local.
> 3. Cập nhật mọi số có gắn 🔸 bên dưới bằng số liệu full-eval mới.
> 4. Điền **Họ tên/MSSV**, và tự viết `submission/REFLECTION.md` (phần đó không điền thay được).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 *(results/runs.csv)* |

**`max_length` có khớp p95 không?** Không hoàn toàn — p95 đo được chỉ 98 token, NB1 gợi ý
`max_length=256` là đủ, nhưng tier T4 giữ mặc định 1024. Không sai (1024 > 98 nên không cắt
mất câu trả lời nào), nhưng dư thừa: mỗi batch đang pad tới 1024 trong khi 98 là đủ, làm
train/eval chậm hơn và tốn VRAM activation hơn cần thiết. Nếu chạy lại, hạ `max_length`
xuống ~128–256 sẽ hợp lý hơn với corpus này.

**Template có giữ khối `<think>` không?** **Có** — verdict trong `template_check.json`:
*"reasoning preserved — safe to train on traces"*. Khối `<think></think>` rỗng vẫn được
giữ nguyên trong chat template khi render prompt sinh, nên không cần xử lý gì thêm.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (từ `results/mask_proof.json`, trường `supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

`supervised_fraction = 0.41 < 0.95` → không tính loss trên prompt, đạt yêu cầu 1.1.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train) 🔸 *(n=8, smoke — cần chạy lại full)*

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.75 | 0.00 | 3678.9 |
| (b) base + optimized prompt | 0.6875 | 0.75 | 1.00 | 1034.6 |
| (c) LoRA fine-tune | 0.9375 | 0.75 | 1.00 | 1519.7 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — target nhảy từ 0.000 → 0.688, format từ
0.00 → 1.00 (naive prompt không ép được model trả JSON hợp lệ), và latency còn giảm gần
3.6x (naive prompt khiến model sinh thêm văn bản giải thích thừa). Không sửa
`OPTIMIZED_PROMPT` — `make verify` xác nhận "baseline (b) prompt unmodified"
(SHA `719e74d3b6232053`).

---

## 4. Giải phẫu cấu hình sai (NB4) 🔸 *(cột target đo với n=8 — cần chạy lại full)*

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6268 | 0.9375 | 403.8 | 8.78 |
| `attn_only` | q,v *(matched r=283)* | 283 | 32,456,704 | 1e-4 | 0.5369 | 0.9375 | 267.3 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 398.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.8438 | 469.3 | 3.86 |

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng,
thua, hay hoà?** `attn_only` dùng 32,456,704 tham số so với 32,464,896 của `correct` —
lệch 0,025%, đạt yêu cầu "khớp ngân sách <5%". Trên target cả hai **hoà** (0.9375 = 0.9375).
Thứ tự đó **không khớp hoàn toàn** với train loss: `attn_only` có loss thấp hơn
(0.5369 < 0.6268) nhưng target không vượt lên, chỉ hoà. Điều này gợi ý ở bài toán này,
trong ngân sách step/tham số đã cho, **rank (ngân sách tham số) là đòn bẩy chính hơn vị trí
gắn adapter** — chỉ gắn vào `q,v` với rank đủ lớn cho cùng số tham số đã đạt được độ chính
xác tương đương gắn `all-linear`. Cảnh báo: n=8 (smoke) nên một tie 0.9375=0.9375 có thể
đổi khi chạy full 50 mẫu — cần xác nhận lại.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao?** LR hạ 10x
(1e-5, thang full fine-tune) khiến train loss cuối cao nhất trong 4 run (1.5702, gần gấp
2.5x `correct`) — LoRA gần như không học được gì trong 30 step. Hệ quả nặng hơn con số
loss cho thấy: target rơi về 0.0000 **và** format cũng về 0.00 (model không còn sinh ra
JSON hợp lệ), latency phình to bất thường (5212.5ms so với ~1500-1800ms các run khác) —
dấu hiệu model sinh văn bản dài, lan man, không dừng đúng token. **Nếu chỉ nhìn đường loss
mà không biết LR sai, sẽ dễ kết luận nhầm "train chưa đủ step, cần train lâu hơn"** — thay
vì nhận ra bước cập nhật LoRA quá nhỏ để thay đổi hành vi model, một lỗi về *thang* siêu
tham số chứ không phải thiếu thời gian huấn luyện.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** Tiết kiệm ~4.92 GB VRAM
(8.78 → 3.86 GB, ~56%). Trả giá: target giảm từ 0.9375 → 0.8438 (giảm ~10% tương đối),
train loss cao hơn (0.7058 vs 0.6268), và chậm hơn về wall-clock (469.3s vs 403.8s, +16%,
do overhead dequantize). Số đo này **ủng hộ** khuyến nghị của nhà cung cấp "không dùng
QLoRA cho Qwen3.5" — ở quy mô 4B, bf16 LoRA đã vừa thoải mái trong VRAM T4, nên phần VRAM
QLoRA tiết kiệm được không đáng đánh đổi độ chính xác/tốc độ ở đây. QLoRA vẫn có lý do tồn
tại nếu VRAM mới là ràng buộc cứng (ví dụ cố chạy model 9B trên T4).

---

## 5. Phán quyết (NB5) 🔸 *(đo với n=8 — cần xác nhận lại trên full eval)*

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.00`

Diễn giải: Fine-tune vượt baseline (b) (prompt đã tối ưu) +0.250 điểm target tuyệt đối
(0.9375 so với 0.6875), trong khi **không** làm giảm điểm regression (15 câu hỏi kiến thức
phổ thông — delta đúng 0.000). Điều này quan trọng vì nó loại trừ khả năng điểm target tăng
là do model "quá khớp" vào format task đến mức phá hỏng năng lực tổng quát — cải thiện ở
đây có vẻ là cải thiện thật, cục bộ vào đúng tác vụ triage, không phải đánh đổi năng lực
khác. `valid_trace_rate=0.00` là kỳ vọng đúng (không phải lỗi) vì corpus mặc định không có
khối `<think>` thật trong câu trả lời (ghi chú trong `.env.example`). Phán quyết PASSED này
đáng tin về mặt liêm chính: SHA của prompt (b) không đổi, checksum eval set không đổi — nên
không phải do làm yếu baseline hay sửa eval sau khi thấy kết quả. Giới hạn lớn nhất hiện tại
là **n=8** (mẫu quá nhỏ để chắc chắn) — cần chạy lại với full 50 mẫu target để phán quyết này
đứng vững.

---

## 6. Định tính — bắt buộc có cả ca THUA 🔸 *(8 mẫu smoke; nên bổ sung thêm khi có full eval)*

> Lưu ý: pipeline NB5 chỉ export dự đoán của (c) fine-tune vào `qualitative.json`, không lưu
> dự đoán per-item của (b) — cột "(b) prompt" dưới đây để trống vì chưa có dữ liệu thô,
> không phải bị bỏ qua.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "...đặt chuột không dây... Cho tôi trả lại. Gấp..." | intent=doi_tra, urgency=cao, sentiment=tich_cuc | *(chưa lưu)* | khớp cả 4 trường | ✅ FT thắng |
| 2 | "...đặt balo laptop... Đổi size. Hỏi cho biết thôi..." | intent=doi_tra, urgency=thap, sentiment=tieu_cuc | *(chưa lưu)* | khớp cả 4 trường | ✅ FT thắng |
| 3 | "...đặt đèn bàn LED... Hoàn tiền. Quá hạn rồi..." | intent=hoan_tien, urgency=cao, sentiment=tich_cuc | *(chưa lưu)* | khớp cả 4 trường | ✅ FT thắng |
| 4 | "...đặt bình giữ nhiệt... Chưa thấy tiền. Khi nào tiện..." | urgency=**thap** | *(chưa lưu)* | urgency=**trung_binh** (sai) | ❌ **FT thua** |
| 5 | "...đặt nồi chiên không dầu... Thiếu phụ kiện. Khi nào tiện..." | urgency=**thap** | *(chưa lưu)* | urgency=**trung_binh** (sai) | ❌ **FT thua** |

**Có mẫu chung nào ở các ca FT thua không?** Có — cả 2 ca thua đều sai đúng field
`urgency`, và đều sai theo cùng một hướng: nhãn thật là `thap` (ticket chứa cụm "khi nào
tiện" — tín hiệu urgency thấp gián tiếp, không có từ khóa rõ như "gấp"), nhưng model đoán
`trung_binh`. Có vẻ model học được các tín hiệu urgency **tường minh** (ví dụ "gấp" → cao)
tốt hơn tín hiệu **ngầm định** (câu không nói gì đặc biệt → model mặc định về mức trung
bình an toàn thay vì nhận ra đó là dấu hiệu của mức thấp).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Với số liệu hiện có (lưu ý: đo trên n=8, cần xác nhận lại trên full-eval),
bản fine-tune LoRA `correct` thắng rõ baseline đã prompt tối ưu (+0.250 target, không mất
regression) và cổng hồi quy PASS — nên **có cơ sở để cân nhắc deploy**, nhưng nên làm vậy
sau khi xác nhận lại trên tập eval đầy đủ (50 mẫu), vì khoảng cách 0.9375 vs 0.6875 ở n=8 có
biên dao động lớn. Đòn bẩy thật sự trong lab này, dựa trên NB4, **không phải vị trí gắn
adapter** — `attn_only` (chỉ 2 module q,v) đạt target ngang `correct` (12 module) khi đã
khớp ngân sách tham số (rank nâng lên 283 để bù lại ít module hơn). Đòn bẩy rõ ràng nhất là
**thang learning rate**: hạ LR 10x (`wrong_lr`) phá hỏng hoàn toàn cả target và format, tệ
hơn cả việc đổi vị trí hay lượng tử hoá. Mask đúng (`supervised_fraction=0.41`, câu hỏi
không bị tính loss) là điều kiện cần để mọi so sánh trên có nghĩa — nếu mask sai từ NB1,
không con số nào ở các bước sau còn đáng tin.

**Ba điều tôi học được** *(gợi ý — nên viết lại bằng trải nghiệm thật của bạn khi chạy
pipeline, phần này chấm theo tính cá nhân/cụ thể)*:
1.
2.
3.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
