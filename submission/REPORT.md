# Lab 21 — Evaluation Report

**Họ tên**: Lê Gia Bảo  **MSSV**: 2A202602887  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 (~14.6 GB khả dụng), precision fp16`

> Mọi số liệu trong report này lấy từ **full eval set** (`results/*.json`,
> `n_target=50`, `n_regression=15`, `smoke_mode=false`) — không còn là bản rút gọn
> `EVAL_LIMIT=8`. Nên đọc lại §7 "Ba điều tôi học được" và `submission/REFLECTION.md`
> trước khi nộp — cả hai được soạn dựa trên các sự kiện thật đã xảy ra trong lúc làm lab
> này, nhưng vẫn nên sửa lại bằng đúng lời của bạn trước khi coi là bản cuối.

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

## 3. Ba baseline (NB2 — đo TRƯỚC khi train, full eval n=50 target / n=15 regression)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.00 | 3172.2 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.00 | 1002.0 |
| (c) LoRA fine-tune | 0.9700 | 0.5444 | 1.00 | 1377.9 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — target nhảy từ 0.000 → 0.765, format từ
0.00 → 1.00 (naive prompt gần như không bao giờ ép được model trả JSON hợp lệ), và latency
giảm hơn 3x (naive prompt khiến model sinh thêm văn bản giải thích thừa, dài và chậm hơn).
Không sửa `OPTIMIZED_PROMPT` — `make verify` xác nhận "baseline (b) prompt unmodified"
(SHA `719e74d3b6232053`).

**Nhìn thoáng (c) có vẻ thắng cả (a) và (b) ở target (0.97) — nhưng xem mục 5, cột
`regression` của (c) đã tụt từ 0.7911 xuống 0.5444. Đây là trọng tâm thật của report này.**

---

## 4. Giải phẫu cấu hình sai (NB4) — full eval, n=50 target

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6268 | 0.970 | 403.8 | 8.78 |
| `attn_only` | q,v *(matched r=283)* | 283 | 32,456,704 | 1e-4 | 0.5369 | 0.965 | 267.3 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 398.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 469.3 | 3.86 |

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng,
thua, hay hoà?** `attn_only` dùng 32,456,704 tham số so với 32,464,896 của `correct` —
lệch 0,025%, đạt yêu cầu "khớp ngân sách <5%". Trên target đầy đủ (50 mẫu), `attn_only`
đạt 0.965 so với 0.970 của `correct` — khác đúng **1 mẫu / 50**, về bản chất là **hoà**
trong biên nhiễu của một tập 50 mẫu. Thứ tự đó **ngược** với train loss: `attn_only` có
loss thấp hơn hẳn (0.5369 so với 0.6268 — tốt hơn ~14%) nhưng target lại nhích thấp hơn một
chút, không vượt lên. Điều đó nói rất rõ: ở ngân sách tham số và số step đã cho, **rank
(ngân sách tham số) là đòn bẩy chính, không phải vị trí gắn adapter** — chỉ gắn vào 2 module
`q,v` với rank đủ lớn để bù lại đã đạt độ chính xác downstream gần như y hệt gắn vào cả 12
module `all-linear`. Train loss thấp hơn không đồng nghĩa với target cao hơn — một ví dụ cụ
thể cho việc không nên xếp hạng bằng loss (Lỗi #3 trong rubric).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao?** LR hạ 10x
(1e-5, thang full fine-tune) khiến train loss cuối cao nhất trong 4 run (1.5702, gần gấp
2.5x `correct`) — LoRA gần như không học được gì trong 30 step. Hệ quả nặng hơn con số loss
cho thấy: target rơi về **0.000** và format cũng về **0.00** (model không còn sinh ra JSON
hợp lệ), latency phình to bất thường (5130.8ms so với ~1400-1700ms các run khác) — dấu hiệu
model sinh văn bản dài, lan man, không dừng đúng token. **Nếu chỉ nhìn đường loss mà không
biết LR sai, sẽ dễ kết luận nhầm "train chưa đủ step, cần train lâu hơn"** — thay vì nhận ra
bước cập nhật LoRA quá nhỏ để thay đổi hành vi model, một lỗi về *thang* siêu tham số chứ
không phải thiếu thời gian huấn luyện.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** Tiết kiệm ~4.92 GB VRAM
(8.78 → 3.86 GB, ~56%). Trả giá: target giảm từ 0.970 → 0.940 (giảm 3 điểm phần trăm tuyệt
đối — nhỏ hơn tôi ước tính ban đầu từ dữ liệu smoke), train loss cao hơn (0.7058 vs 0.6268),
và chậm hơn về wall-clock (469.3s vs 403.8s, +16%, do overhead dequantize). Số đo này **ủng
hộ nhưng không áp đảo** khuyến nghị của nhà cung cấp "không dùng QLoRA cho Qwen3.5": ở quy
mô 4B, bf16 LoRA đã vừa thoải mái trong VRAM T4 nên không cần đánh đổi; nhưng cái giá đo
được (3pp target, +16% thời gian) là khá nhỏ — nếu VRAM là ràng buộc cứng (ví dụ model 9B
trên T4), QLoRA vẫn là lựa chọn hợp lý.

---

## 5. Phán quyết (NB5) — full eval, n=50 target / n=15 regression

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.00`

Diễn giải: Fine-tune vượt baseline (b) (prompt đã tối ưu) **+0.205** điểm target tuyệt đối
(0.970 so với 0.765) — một cải thiện thật, không phải nhiễu, và không phải do làm yếu (b)
(SHA prompt không đổi, (b) > (a) như mục 3 đã xác nhận). Nhưng đồng thời, điểm **regression**
(15 câu hỏi kiến thức/chỉ dẫn phổ thông, đo trên cùng model) sụt từ 0.7911 xuống **0.5444**
— giảm **24.7 điểm phần trăm**, vượt xa ngưỡng chấp nhận 2.0pp mà cổng hồi quy đặt ra. Đây
là **catastrophic forgetting** điển hình: train 250 mẫu hẹp (chỉ một tác vụ JSON-triage)
trong 2 epoch đã kéo trọng số model lệch đủ nhiều để phá hỏng năng lực tổng quát, dù target
task học rất tốt. Lý do kỹ thuật phía sau (ghi trong `verdict.json`): corpus huấn luyện
không trộn dữ liệu phổ thông để "giữ chỗ" (deck §6.3 gợi ý trộn 1–5% dữ liệu tổng quát).
Phán quyết FAILED này đáng tin cậy về mặt liêm chính (checksum eval không đổi, prompt (b)
không bị làm yếu) — và theo đúng kết luận đúng của phần này: **bản fine-tune `correct` hiện
tại KHÔNG nên deploy**, vì cái giá về năng lực tổng quát lớn hơn lợi ích accuracy triage thu
được, đối với một trợ lý cần xử lý cả yêu cầu ngoài phạm vi triage.

---

## 6. Định tính — bắt buộc có cả ca THUA (full eval, 50 mẫu)

> Lưu ý: pipeline NB5 chỉ export dự đoán của (c) fine-tune vào `qualitative.json`, không lưu
> dự đoán per-item của (b) — cột "(b) prompt" dưới đây để trống vì chưa có dữ liệu thô,
> không phải bị bỏ qua.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "...đặt chuột không dây... Cho tôi trả lại. Gấp..." | intent=doi_tra, urgency=cao, sentiment=tich_cuc | *(chưa lưu)* | khớp cả 4 trường | ✅ FT thắng |
| 2 | "...đặt balo laptop... Đổi size. Hỏi cho biết thôi..." | intent=doi_tra, urgency=thap, sentiment=tieu_cuc | *(chưa lưu)* | khớp cả 4 trường | ✅ FT thắng |
| 3 | "...đặt bình giữ nhiệt... Chưa thấy tiền. **Khi nào tiện**..." | urgency=**thap** | *(chưa lưu)* | urgency=**trung_binh** (sai) | ❌ **FT thua** |
| 4 | "...đặt nồi chiên không dầu... Thiếu phụ kiện. **Khi nào tiện**..." | urgency=**thap** | *(chưa lưu)* | urgency=**trung_binh** (sai) | ❌ **FT thua** |
| 5 | "...đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. **Khi nào tiện**..." | urgency=**thap** | *(chưa lưu)* | urgency=**trung_binh** (sai) | ❌ **FT thua** |

**Có mẫu chung nào ở các ca FT thua không?** Có, và rất rõ ràng trên toàn bộ 50 mẫu: tất cả
**6/6** ticket bị FT chấm sai (điểm 0.75/1.0, chiếm 12% tập target) đều sai đúng cùng một
field (`urgency`), đều sai theo cùng một hướng (nhãn thật `thap`, model đoán `trung_binh`),
và đều là ticket chứa đúng cụm từ **"Khi nào tiện"** làm tín hiệu urgency — một tín hiệu
*ngầm định/lịch sự* thay vì từ khóa rõ ràng như "gấp" (urgency cao) hay không có tín hiệu gì
(thường là urgency thấp/trung bình mặc định). Model dường như học tốt các tín hiệu urgency
**tường minh** nhưng chưa học được rằng cụm "khi nào tiện" cụ thể này là dấu hiệu của mức
**thấp** — một lỗi hệ thống, không phải ngẫu nhiên, và là manh mối tốt để cải thiện dữ liệu
huấn luyện (thêm nhiều biến thể của cụm "khi nào tiện" gắn nhãn `thap` vào training set).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Bản fine-tune LoRA `correct` **không nên deploy ở trạng thái hiện tại**.
Lý do nhân quả: train 250 mẫu hẹp, chỉ một tác vụ (JSON triage), trong 2 epoch, không trộn
dữ liệu tổng quát — khiến model học rất tốt tác vụ hẹp (target 0.765→0.970, +20.5pp so với
baseline đã prompt tối ưu) nhưng trả giá bằng việc quên năng lực tổng quát (regression
0.7911→0.5444, −24.7pp), vượt xa ngưỡng chấp nhận của cổng hồi quy (2pp) — đây là
catastrophic forgetting, không phải nhiễu đo đạc, vì cả hai con số đều đo trên cùng tập đã
đóng băng từ NB2. Đòn bẩy thật sự lộ ra qua NB4 **không phải vị trí gắn adapter**: `attn_only`
(chỉ 2 module q,v, rank nâng lên 283 để khớp ngân sách tham số) đạt target gần như y hệt
`correct` (0.965 vs 0.970, khác 1/50 mẫu) dù train loss thấp hơn — tức rank/ngân sách tham
số quan trọng hơn việc chọn module nào để gắn LoRA, ít nhất ở quy mô 32M tham số huấn luyện
này. Ngược lại, learning rate sai thang (`wrong_lr`, −10x) là lỗi nặng nhất đo được — phá
sập cả target và format về 0, nặng hơn cả hai lỗi cấu hình khác gộp lại. Mask đúng
(`supervised_fraction=0.41`, câu hỏi không bị tính loss — NB1) là điều kiện nền cho mọi so
sánh ở trên có nghĩa; nếu mask sai, không con số nào trong 6 mục trước còn đáng tin. Hướng
sửa rõ ràng nhất cho lần train sau: trộn 1–5% dữ liệu instruction phổ thông vào tập 250 mẫu
(đúng gợi ý trong `verdict.json` và deck §6.3), rồi đo lại đúng cổng hồi quy này.

**Ba điều tôi học được** *(gợi ý dựa trên số liệu — nên viết lại/bổ sung bằng trải nghiệm
thật của bạn khi chạy pipeline, phần này chấm theo tính cá nhân/cụ thể)*:
1. Target accuracy tăng không đồng nghĩa với "nên deploy" — phải luôn nhìn cùng lúc với
   regression; một model tốt hơn ở đúng việc được giao vẫn có thể là một bước lùi nếu nó
   quên mất mọi việc khác.
2. Rank-matched `attn_only` hoà với `all-linear` ở cùng ngân sách tham số khiến tôi phải xem
   lại giả định "gắn nhiều module hơn luôn tốt hơn" — ở quy mô dữ liệu/step nhỏ, ngân sách
   tham số mới là biến quyết định.
3. Train loss là chỉ số dễ đọc nhưng dễ gây hiểu sai nhất trong cả 4 run NB4 — `wrong_lr` và
   `attn_only` đều cho thấy thứ tự loss không phải lúc nào cũng là thứ tự chất lượng thật.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu instruction phổ thông vào training
set rồi train lại `correct`, đo lại cổng hồi quy để xem regression có về lại trong ngưỡng
không, trong khi vẫn giữ được phần lớn +20.5pp target đã đạt được.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
