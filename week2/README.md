# Bài tập Phân loại cảm xúc bình luận phim IMDB - Week 2

## Giới thiệu

Bài tập này xây dựng các mô hình deep learning để phân loại cảm xúc (tích cực hoặc tiêu cực) từ các đánh giá phim trên IMDB. Mô hình được huấn luyện trên tập dữ liệu IMDB Movie Reviews với 5000 mẫu cho tập huấn luyện và 5000 mẫu cho tập kiểm thử.

## Cấu trúc bài tập

```
.
├── dataset/               # Thư mục chứa dữ liệu IMDB
│   ├── IMDB_Dataset.csv   # Dữ liệu gốc
│   ├── vocab.pkl          # Từ điển được xây dựng từ dữ liệu
│   └── processed_data.pkl # Dữ liệu đã qua xử lý
├── checkpoint/            # Thư mục lưu trữ các checkpoint và kết quả
│   ├── lstm_best.pt       # Model LSTM tốt nhất
│   ├── cnn_best.pt        # Model CNN tốt nhất
│   ├── bilstm_attn_best.pt# Model BiLSTM với Attention tốt nhất
│   └── ...                # Các biểu đồ và kết quả khác
├── data.py                # Module xử lý dữ liệu
├── model.py               # Module định nghĩa và huấn luyện mô hình
├── evaluate.py            # Module đánh giá mô hình
├── config.json            # Cấu hình thử nghiệm
├── results.csv            # Kết quả thử nghiệm các siêu tham số
└── README.md              # File này
```

## Các mô hình được cài đặt

### 1. LSTM

-   Mô hình LSTM cơ bản với word embedding
-   Tùy chọn bidirectional
-   Nhiều lớp LSTM
-   Dropout để chống overfitting

### 2. CNN

-   Mạng CNN cho phân loại văn bản
-   Sử dụng nhiều bộ lọc với các kích thước khác nhau
-   Max-over-time pooling
-   Dropout để chống overfitting

### 3. BiLSTM với Attention

-   Mô hình LSTM hai chiều
-   Thêm cơ chế attention để tập trung vào các từ quan trọng
-   Nhiều lớp BiLSTM
-   Dropout để chống overfitting

## Tiền xử lý dữ liệu

Quá trình tiền xử lý dữ liệu bao gồm các bước:

1. Chuyển về chữ thường
2. Loại bỏ HTML tags
3. Loại bỏ URL
4. Loại bỏ các ký tự đặc biệt và số
5. Tokenization
6. Loại bỏ stopwords
7. Lemmatization
8. Xây dựng từ điển với các từ phổ biến nhất
9. Chuyển đổi văn bản thành chuỗi số
10. Padding và truncating để đạt độ dài cố định

## Cài đặt

1. Clone repository này
2. Cài đặt các thư viện cần thiết:

```bash
pip install torch numpy pandas matplotlib scikit-learn seaborn nltk
```

3. Tải dữ liệu IMDB từ [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) và đặt file `IMDB_Dataset.csv` vào thư mục `dataset/`

## Sử dụng

### 1. Xử lý dữ liệu

```bash
python data.py
```

### 2. Huấn luyện mô hình

```bash
python model.py
```

Mặc định sẽ huấn luyện 3 loại mô hình (LSTM, CNN, BiLSTM với Attention) với cấu hình từ file `config.json`.

### 3. Đánh giá mô hình

```bash
python evaluate.py
```

## Thử nghiệm siêu tham số

Các siêu tham số được thử nghiệm bao gồm:

-   Batch size: 16, 32, 64, 128
-   Learning rate: 0.001, 0.01, 0.0001
-   Số lớp ẩn: 1, 2, 3
-   Số nơron trong lớp ẩn: 128, 256, 512
-   Kích thước embedding: 100, 200, 300
-   Optimizer: Adam, RMSprop, SGD
-   Các kiến trúc mạng: LSTM, CNN, BiLSTM với Attention

Mỗi cấu hình được chạy 3 lần để tính trung bình và độ lệch chuẩn của các metrics.

## Kết quả

Kết quả chi tiết được lưu trong file `results.csv` và các biểu đồ trong thư mục `checkpoint/`.

Các metrics đánh giá bao gồm:

-   Accuracy
-   Precision
-   Recall
-   F1 score

## Tác giả

Đỗ Hoàng Vũ

## Giấy phép

MIT License
