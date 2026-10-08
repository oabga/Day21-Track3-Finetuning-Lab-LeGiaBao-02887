# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Lần chạy đầu (NB2+NB5 với `EVAL_LIMIT=8` — mặc định sẵn trong ô 3 của
`Lab21_RUN_ALL.ipynb`, tôi không để ý nên chạy nguyên mặc định) cho verdict **PASSED**,
target thắng baseline (b) +0.250. Tưởng đã xong. Chạy lại với tập eval đầy đủ (50 mẫu
target, 15 mẫu regression) thì verdict lật thành **FAILED**: target vẫn thắng
(+0.205), nhưng điểm regression (15 câu hỏi kiến thức phổ thông) tụt từ 0.791 xuống
0.544 — giảm gần 25 điểm phần trăm, vượt xa ngưỡng cho phép. Với 8 mẫu, mọi chuyện nhìn
rất ổn; với 50 mẫu, lộ ra rõ model đã quên mất gần một nửa năng lực tổng quát. Ngạc
nhiên nhất là khoảng cách giữa "nhìn có vẻ ổn" và "thật sự ổn" lớn đến mức nào, chỉ vì
một tham số mặc định tôi không để ý.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở phần train hay hiểu lý thuyết LoRA/rank/learning rate — phần đó đọc
`runs.csv` và `autopsy.json` là ra ngay. Mất thời gian nhất là ở việc **tải file từ
Colab về máy** (right-click vào folder không có nút Download, phải dùng
`shutil.make_archive` + `files.download()`, rồi còn bị trình duyệt chặn pop-up) và việc
phát hiện ra mình đã chạy nhầm chế độ rút gọn `EVAL_LIMIT=8` nên phải quay lại Colab
chạy lại đúng NB2+NB5. Tôi không dự đoán trước được hai điểm này — nghĩ khó nhất sẽ là
phần cấu hình LoRA, hoá ra lại là cơ chế vận hành (thao tác Colab, đọc kỹ tham số mặc
định trước khi bấm Run).

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây nghĩ target accuracy tăng là đủ để nói "fine-tune thành công". Sau khi thấy
`correct` thắng target +20.5 điểm phần trăm nhưng đồng thời phá sập regression −24.7
điểm phần trăm (catastrophic forgetting, vì 250 mẫu train chỉ có một tác vụ hẹp, không
trộn dữ liệu tổng quát), tôi hiểu một con số tăng không nói lên gì nếu không đặt cạnh
con số có thể bị nó làm hỏng. Cũng từng nghĩ gắn LoRA vào nhiều module (`all-linear`)
chắc luôn tốt hơn gắn ít module (`q,v` only) — nhưng khi khớp đúng ngân sách tham số
(rank nâng lên để bù), `attn_only` cho target gần như y hệt `correct` (0.965 vs 0.970).
Rank/ngân sách tham số mới là cái quyết định, không phải việc chọn gắn vào đâu.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng Claude Code để: đọc hiểu README/rubric, setup môi trường CPU local, chạy NB1 và
`make verify`, hướng dẫn từng bước chạy Colab, xử lý lỗi tải file, và viết
`submission/REPORT.md` dựa trên số liệu thật trong `results/`. Chỗ nó đúng mà tôi cần
ghi nhận: nó từ chối bịa số liệu/tên/trải nghiệm cá nhân khi tôi yêu cầu "làm tạm cho
xong" và "làm sao để không bị phát hiện" — đúng lúc tôi đang muốn đi tắt. Chỗ cần cẩn
thận: bản report đầu tiên nó viết là từ dữ liệu `EVAL_LIMIT=8` (verdict PASSED) — không
sai dữ liệu, nhưng nếu tôi không chạy lại full-eval mà nộp luôn bản đó thì kết luận
trong report đã là kết luận sai (PASSED thay vì FAILED thật).

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng một tập eval đủ lớn gồm **cả** tập đo đúng tác vụ khách hàng cần **và** một
tập đo năng lực tổng quát/không liên quan, trước khi train — và quyết định trước ngưỡng
chấp nhận được cho tập tổng quát (như cổng hồi quy của lab này). Lab này cho thấy nếu
không có tập đối chứng đó, rất dễ nộp một model "thắng" trên đúng chỉ số được yêu cầu
nhìn, mà không biết nó đã phá hỏng gì khác.
