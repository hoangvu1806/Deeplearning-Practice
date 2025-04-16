# Phân loại Cảm xúc Đánh giá Phim IMDb

Dự án này xây dựng mô hình học sâu sử dụng PyTorch để phân loại cảm xúc (tích cực/tiêu cực) của đánh giá phim từ tập dữ liệu IMDb Movie Reviews.

## Giới thiệu

Phân loại cảm xúc (Sentiment Analysis) là một bài toán xử lý ngôn ngữ tự nhiên quan trọng, giúp máy tính hiểu được thái độ, cảm xúc hay quan điểm của con người thông qua văn bản. Dự án này tập trung vào việc phân loại đánh giá phim thành hai loại: tích cực hoặc tiêu cực, sử dụng các kiến trúc mạng nơ-ron hồi quy như RNN và LSTM.

## Cấu trúc Dự án

```
/Week2
│
├── main.py              # Script chính để huấn luyện mô hình
├── models.py            # Định nghĩa các mô hình RNN và LSTM
├── custom_models.py     # Các mô hình tự xây dựng (nhưng chưa dùng)
├── preprocess_data.py   # Tiền xử lý dữ liệu
├── evaluate.py          # Đánh giá và trực quan hóa kết quả
├── report.md            # Báo cáo kết quả và nhận xét
│
├── dataset/             # Thư mục chứa dữ liệu
│   └── processed_data.pkl  # Dữ liệu đã tiền xử lý
│   └── IMDB_Dataset.csv  # Dữ liệu gốc
│
├── checkpoint/          # Thư mục lưu mô hình đã huấn luyện
│
└── results/             # Kết quả thí nghiệm và hình ảnh
    └── results.json     # Kết quả từ các thí nghiệm
```

## Công nghệ sử dụng

-   **Python 3.8+**
-   **PyTorch**: Framework học sâu chính
-   **NLTK**: Thư viện xử lý ngôn ngữ tự nhiên
-   **matplotlib & seaborn**: Trực quan hóa dữ liệu
-   **scikit-learn**: Đánh giá mô hình

## Tiền xử lý dữ liệu

Quá trình tiền xử lý dữ liệu bao gồm:

-   Tokenization: Tách văn bản thành các từ riêng biệt
-   Chuyển đổi văn bản thành chữ thường
-   Loại bỏ ký tự đặc biệt và dấu câu
-   Xây dựng từ điển (vocabulary)
-   Chuyển đổi các đoạn văn bản thành chuỗi số nguyên
-   Padding để đảm bảo các chuỗi có cùng độ dài

## Mô hình

Dự án triển khai hai kiến trúc mô hình chính:

1. **RNN (Recurrent Neural Network)**: Mạng nơ-ron hồi quy cơ bản
2. **LSTM (Long Short-Term Memory)**: Mạng hồi quy có khả năng ghi nhớ thông tin dài hạn tốt hơn

Mỗi mô hình được huấn luyện với nhiều cấu hình siêu tham số khác nhau:

-   Số lớp ẩn (num_layers)
-   Kích thước lớp ẩn (hidden_size)
-   Kích thước batch (batch_size)
-   Tốc độ học (learning_rate)
-   Loại optimizer (adam, adamw, rmsprop, nadam)

## Thí nghiệm và Đánh giá

Mỗi cấu hình mô hình được huấn luyện 3 lần để đảm bảo kết quả đáng tin cậy. Đánh giá dựa trên:

-   Độ chính xác (accuracy) trên tập kiểm thử
-   Ma trận nhầm lẫn (confusion matrix)
-   Báo cáo phân loại chi tiết (precision, recall, f1-score)

Kết quả được trực quan hóa thông qua:

-   Biểu đồ so sánh độ chính xác giữa các cấu hình
-   Biểu đồ so sánh hiệu suất giữa RNN và LSTM
-   Biểu đồ phân tích ảnh hưởng của các siêu tham số

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

```bash
python preprocess_data.py
```

### 2. Huấn luyện mô hình

```bash
python main.py
```

### 3. Đánh giá và trực quan hóa kết quả

```bash
python evaluate.py
```

### 4. Kiểm tra với đánh giá tùy chỉnh

Chạy `python evaluate.py` và chọn 'y' khi được hỏi về việc kiểm tra đánh giá phim. Sau đó nhập đánh giá phim của bạn để nhận kết quả phân loại.

## Kết quả

LSTM cho hiệu suất tốt hơn RNN trong hầu hết các cấu hình. Cấu hình tốt nhất đạt được độ chính xác khoảng 86.6% trên tập kiểm thử.

Chi tiết kết quả và phân tích có thể được tìm thấy trong file [report.md](report.md) và thư mục [results](results/).
