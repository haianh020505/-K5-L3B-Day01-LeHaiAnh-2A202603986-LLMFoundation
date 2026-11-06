# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603986
- **Họ và tên:** Lê Hải Anh
- **SEED:** 6153

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng kết quả từ Phần 5 của notebook:

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0.924 | 0.974 | 0.068 | 0.0 |
| BoW + MLP | 0.963 | 0.974 | 1.299 | 0.0 |
| Encoder MiniLM + kNN | 0.914 | 0.947 | 129.269 | 0.0 |
| Encoder MiniLM + Logistic Regression | 0.959 | 0.974 | 27.569 | 0.0 |
| Encoder MiniLM + MLP head | 0.959 | 0.974 | 32.935 | 0.0 |
| LLM nhỏ zero-shot | 0.976* | 0.976 | 2465.148 | 0.0** |
| LLM lớn zero-shot | 0.952* | 0.952 | 2447.321 | 0.0** |
| LLM nhỏ few-shot | 1.000* | 1.000 | 3151.240 | 0.0** |

## 3. Dự đoán và thực tế

### 3.1. Dự đoán `ml_f1`

**Dự đoán đã ghi:**

“F1 khoảng 0.85, precision cao hơn recall”

**Kết quả thực tế:**

- F1 test: 0.924.
- Precision: 0.986.
- Recall: 0.869.

F1 thực tế cao hơn dự đoán 0.074. Tuy nhiên, dự đoán precision cao hơn recall là đúng.

**Giải thích:**

Mình đánh giá thấp hiệu quả của BoW vì các từ khóa như “free”, “call”, “txt” vẫn giúp phân loại tốt trên tập test. Recall thấp hơn precision cho thấy mô hình còn bỏ sót spam.

### 3.2. Dự đoán `dl_relu`

**Dự đoán đã ghi:**

“Xóa ReLU, hai lớp Linear gộp thành một lớp tuyến tính thì tương đương Logistic Regression. F1 có thể giảm so với MLP có ReLU và gần với Logistic Regression.”

**Kết quả thực tế:**

- MLP có ReLU: F1 test = 0.963.
- Mạng không ReLU: F1 test = 0.963.
- Logistic Regression ở Phần 1: F1 test = 0.924.

Bỏ ReLU không làm giảm F1 ở độ chính xác ba chữ số được báo cáo.

**Giải thích:**
Bỏ ReLU khiến mạng trở thành tuyến tính, nhưng cách huấn luyện khác nhau vẫn có thể cho kết quả khác nhau. Trong lần chạy này, thêm ReLU chưa giúp tăng F1.st.

### 3.3. Dự đoán `llm_models`

**Dự đoán đã ghi:**

“Model lớn có F1 nhỉnh hơn khoảng 0.02–0.05, chi phí cao hơn vài lần.”

**Kết quả thực tế:**

- Model nhỏ zero-shot: F1 = 0.976.
- Model lớn zero-shot: F1 = 0.952.

Model lớn thấp hơn model nhỏ 0.024 F1, trái với dự đoán.

**Giải thích:**

Model lớn hơn không đảm bảo phân loại tốt hơn. SUB chỉ có 40 tin nên chênh lệch một tin sai cũng làm F1 thay đổi đáng kể;

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 giảm từ 0.974 xuống 0.333, giảm 0.641. Các nhóm khó gồm spam giọng tự nhiên (đúng 10%), thao túng/lừa đảo (khoảng 12%) và né từ khóa (khoảng 17%). Mô hình phụ thuộc vào token quen thuộc nên dễ bỏ sót spam viết tự nhiên hoặc biến dạng từ khóa.  

### Hai lỗi cụ thể:

**Case #2:**

> FREE pizza in the office kitchen right now, come quick before it's gone

Nhãn thật ham, dự đoán spam. Mô hình có thể bị các từ “FREE”, “right now” gây nhầm, bỏ qua ngữ cảnh thông báo nội bộ

**Case #47:**

> lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for this?

Nhãn thật ham, dự đoán spam. Mô hình có thể nhầm việc kể lại một tin rác với việc trực tiếp gửi tin rác.

## 5. Thử thách cá nhân

**SEED:** 6153.

**Đề #4:** Embedding — chạy `knn_predict` với `k = 1, 5, 25`, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

### Cách làm

Giữ nguyên encoder MiniLM, embedding của tập train/test và nhãn dữ liệu. Sau đó, thay lần lượt ba giá trị k, lấy dự đoán từ `knn_predict` và tính F1 dựa trên TP, FP, FN.

### Kết quả

| k | F1 test |
| ---: | ---: |
| 1 | 0.909 |
| 5 | 0.914 |
| 25 | 0.895 |

### Giải thích

Với k = 1, kết quả phụ thuộc hoàn toàn vào một hàng xóm nên dễ bị nhiễu hoặc một mẫu không thực sự phù hợp chi phối.

Với k = 25, mô hình lấy thêm nhiều hàng xóm ít liên quan hơn. Do lớp ham chiếm đa số, việc tăng k còn có thể khiến kết quả nghiêng về ham và bỏ sót spam.

k = 5 cân bằng tốt hơn trong thí nghiệm này. Nếu chọn k cho sản phẩm, cần chọn trên validation rồi đánh giá cuối cùng trên tập test độc lập.

## 6. Phương án cho sản phẩm thật

Mình chọn **MiniLM + Logistic Regression** vì đạt **p95 = 27.569 ms/tin < 50 ms**, **F1 test = 0.959** và **F1 trên 50 case khó = 0.640**, cao hơn hai mô hình BoW.

Throughput cả lô khoảng **1.124 tin/giây**, cao hơn lưu lượng trung bình **116 tin/giây** của 10 triệu SMS/ngày. Tuy nhiên, cần kiểm thử tải thực tế để xác nhận p95 khi có nhiều request đồng thời.

Đánh đổi là tốn tài nguyên và chậm hơn BoW do phải chạy encoder. Khả năng xử lý case khó vẫn hạn chế, nên cần bổ sung dữ liệu có nhãn trước khi triển khai rộng.

## 7. AI log và điều đã học

Bảng dưới đây tổng hợp AI log đã ghi trong notebook và lần hỗ trợ lập báo cáo này.

| Phần | Hỏi AI gì | AI trả lời/viết gì | Sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 3 | Giải thích embedding và cosine | Giải thích chuẩn hóa vector, thứ tự lấy top-k và so sánh mô hình | Đối chiếu kết quả tìm kiếm và bảng F1 |
| Phần 4 | Hoàn thiện TODO gọi API, prompt, parse nhãn và batch | Viết code các hàm, hướng dẫn sửa lỗi gọi API | Chạy ô kiểm tra và thí nghiệm; notebook ghi nhận các kiểm tra đạt |
| Phần 5 | Giải thích Pareto, stress test và cách xây dựng case khó | Giải thích đường Pareto, SLA, phân tích lỗi theo nhóm và đề xuất 5 case mới | Đối chiếu biểu đồ và bảng kết quả; chưa có output đánh giá lại 5 case mới |
| Phản ánh cuối bài | Chọn mô hình triển khai, đề xuất router và rút ra bài học | Gợi ý so sánh mô hình và thiết kế router | Đối chiếu số liệu; phần chi phí trong notebook dùng giả định của bài |
| Báo cáo cá nhân | Đọc notebook và lập báo cáo theo mẫu Markdown | Tổng hợp kết quả, phân tích dự đoán, case khó và phương án triển khai | AI đối chiếu trực tiếp code và output; cần đọc lại, xác nhận và bổ sung bằng chứng trước khi nộp |

### Điều đã học

Mình hiểu rằng F1 cao trên tập cùng phân phối với train chưa chứng minh mô hình xử lý tốt các tình huống mới, thể hiện ở việc BoW + LogReg giảm từ 0.974 trên SUB xuống 0.333 ở bộ khó. Mô hình lớn hơn hoặc kiến trúc phức tạp hơn không tự động cho kết quả tốt hơn, mà phải kiểm chứng bằng thí nghiệm. Khi chọn mô hình triển khai, cần xét đồng thời chất lượng, độ trễ và chi phí thực tế, đồng thời phân biệt chi phí chưa được tính với chi phí bằng 0.
