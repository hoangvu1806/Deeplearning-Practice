# Phân loại các bài báo Tiếng Việt với LSTM + Attention
## Sinh viên: Đỗ Hoàng Vũ
## Giảng viên hướng dẫn: Ths. Phạm Xuân Trí
Dự án này xây dựng và đánh giá một mô hình học sâu sử dụng LSTM hai chiều kết hợp với cơ chế Attention để phân loại các bài báo tiếng Việt vào các chủ đề khác nhau. Dự án sử dụng PyTorch và tích hợp Weights & Biases (Wandb) để theo dõi và trực quan hóa quá trình huấn luyện cũng như so sánh các cấu hình siêu tham số.

## Tính năng

-   **Tiền xử lý dữ liệu:** Chuẩn hóa Unicode, làm sạch văn bản (loại bỏ HTML, URL, ký tự đặc biệt), tách từ tiếng Việt (sử dụng `underthesea`).
-   **Mô hình LSTM + Attention:** Sử dụng mô hình LSTM hai chiều để nắm bắt ngữ cảnh và cơ chế Attention để tập trung vào các phần quan trọng của văn bản.
-   **Huấn luyện:** Cung cấp script để huấn luyện mô hình với tùy chọn sử dụng dữ liệu gốc hoặc dữ liệu đã tiền xử lý.
-   **Đánh giá:** Cung cấp script để đánh giá mô hình trên tập kiểm tra, tính toán các chỉ số (accuracy, F1-score), vẽ confusion matrix và phân tích attention weights.
-   **Dự đoán:** Cung cấp script để dự đoán chủ đề cho một văn bản mới (từ text hoặc file) và tùy chọn trực quan hóa attention weights.
-   **Thử nghiệm siêu tham số:** Cung cấp script (`main.py`) để tự động huấn luyện và đánh giá nhiều cấu hình siêu tham số khác nhau, chạy mỗi cấu hình nhiều lần, tính toán kết quả trung bình/độ lệch chuẩn và log lên Wandb.
-   **Tích hợp Wandb:** Theo dõi loss, accuracy, F1-score, learning rate, gradient norm, thông số mô hình trong quá trình huấn luyện và so sánh hiệu suất giữa các lần chạy/cấu hình.

## Cấu trúc thư mục

```
.
├── data/                     # Chứa dữ liệu gốc (cần tự tạo)
│   ├── mapping_data.json     # File ánh xạ đường dẫn file và nhãn
│   └── news/                 # Thư mục chứa các file .txt theo từng chủ đề
│       ├── the-thao/
│       ├── phap-luat/
│       ├── thoi-su/
│       └── ...
├── models/                   # Định nghĩa mô hình PyTorch
│   ├── __init__.py
│   └── lstm_attention.py
├── utils/                    # Các hàm phụ trợ
│   ├── __init__.py
│   ├── dataset.py            # Xử lý Dataset, DataLoader
│   ├── metrics.py            # Hàm tính toán chỉ số, EarlyStopping
│   └── preprocessing.py      # Hàm tiền xử lý văn bản, Vocabulary
├── processed_data/           # Chứa dữ liệu đã tiền xử lý (tạo bởi preprocess_data.py)
│   ├── train_data.pkl
│   ├── val_data.pkl
│   ├── test_data.pkl
│   ├── vocab.pkl
│   └── dataset_info.json
├── saved_models/             # Lưu model, vocab, history từ train.py/evaluate.py
│   ├── best_model.pt
│   ├── vocab.txt             # (Nếu huấn luyện với dữ liệu thô)
│   ├── training_history.json
│   ├── confusion_matrix.png
│   └── attention_analysis/   # Phân tích attention từ evaluate.py
├── experiments/              # Lưu kết quả chi tiết của từng lần chạy từ main.py
│   └── config_X/
│       └── run_Y/
│           ├── best_model.pt
│           └── training_history.json
│           └── training_curves.png
├── results/                  # Lưu kết quả tổng hợp và biểu đồ so sánh từ main.py
│   ├── detailed_results.json
│   ├── summary_results.json
│   ├── summary_results.csv
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   └── error_comparison.png
├── config.py                 # File cấu hình các tham số và đường dẫn
├── main.py                   # Script chính để chạy thử nghiệm siêu tham số
├── preprocess_data.py        # Script tiền xử lý dữ liệu
├── train.py                  # Script huấn luyện mô hình đơn lẻ
├── evaluate.py               # Script đánh giá mô hình
├── predict.py                # Script dự đoán cho văn bản mới
├── requirements.txt          # Danh sách các thư viện cần thiết
├── README.md                 # File này
└── report.md                 # Báo cáo phân tích kết quả thử nghiệm siêu tham số
```

## Cài đặt

1.  **Clone repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Tạo môi trường ảo (khuyến nghị):**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt thư viện:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Chuẩn bị dữ liệu:**

    -   Tạo thư mục `data/news/`.
    -   Trong `data/news/`, tạo các thư mục con tương ứng với từng chủ đề (ví dụ: `the-thao`, `phap-luat`, `giao-duc`,...).
    -   Đặt các file `.txt` chứa nội dung bài báo vào các thư mục chủ đề tương ứng.
    -   Tạo file `data/mapping_data.json` theo định dạng sau, liệt kê đường dẫn tương đối đến từng file và nhãn (tên thư mục) của nó:
        ```json
        [
            {
                "file_path": "data/news/the-thao/article1.txt",
                "category": "the-thao"
            },
            {
                "file_path": "data/news/phap-luat/article2.txt",
                "category": "phap-luat"
            }
            // ... thêm các file khác
        ]
        ```

5.  **Đăng nhập Weights & Biases:**
    -   Nếu bạn chưa có tài khoản, đăng ký tại [https://wandb.ai/](https://wandb.ai/).
    -   Chạy lệnh sau và làm theo hướng dẫn để đăng nhập (cần thiết cho `main.py`):
        ```bash
        wandb login
        ```

## Sử dụng

### 1. Tiền xử lý dữ liệu (Khuyến nghị chạy trước)

Script này sẽ đọc dữ liệu gốc từ `data/`, thực hiện tiền xử lý, xây dựng từ điển, chia tập train/val/test và lưu kết quả vào thư mục `processed_data/` dưới dạng file `.pkl` và `.json`. Việc này giúp tăng tốc độ đáng kể cho các lần chạy huấn luyện/đánh giá/dự đoán sau này vì không cần xử lý lại dữ liệu.

```bash
python preprocess_data.py
```

### 2. Huấn luyện mô hình đơn lẻ (`train.py`)

Script này huấn luyện mô hình với cấu hình được định nghĩa trong `config.py`.

-   **Sử dụng dữ liệu đã xử lý (mặc định):**

    ```bash
    python train.py
    ```

    Script sẽ tự động tìm và sử dụng dữ liệu trong `processed_data/`.

-   **Sử dụng dữ liệu gốc (tiền xử lý trực tiếp):**
    ```bash
    python train.py --raw
    ```
    Script sẽ đọc dữ liệu từ `data/`, tiền xử lý, xây dựng/tải từ điển (`saved_models/vocab.txt`) và huấn luyện.

Kết quả (model tốt nhất `best_model.pt`, lịch sử huấn luyện `training_history.json`, biểu đồ `training_curves.png`) sẽ được lưu vào thư mục `saved_models/`.

### 3. Đánh giá mô hình (`evaluate.py`)

Đánh giá mô hình đã được huấn luyện (mặc định là `saved_models/best_model.pt`) trên tập test.

-   **Sử dụng dữ liệu đã xử lý (mặc định):**

    ```bash
    python evaluate.py
    ```

-   **Sử dụng dữ liệu gốc:**

    ```bash
    python evaluate.py --raw
    ```

    (Cần có file `saved_models/vocab.txt` nếu `train.py --raw` chưa được chạy hoặc vocab chưa được tạo)

-   **Đánh giá model/vocab cụ thể:**
    ```bash
    # Ví dụ đánh giá model từ main.py với dữ liệu gốc
    python evaluate.py --model experiments/config_X/run_Y/best_model.pt --raw --vocab saved_models/vocab.txt
    # Ví dụ đánh giá model từ main.py với dữ liệu đã xử lý
    python evaluate.py --model experiments/config_X/run_Y/best_model.pt
    ```

Script sẽ in ra các chỉ số Accuracy, F1-score, báo cáo chi tiết và lưu confusion matrix (`confusion_matrix.png`), kết quả (`evaluation_results.json`) và phân tích attention (`attention_analysis/`) vào thư mục `saved_models/`.

### 4. Dự đoán thể loại (`predict.py`)

Dự đoán chủ đề cho một văn bản mới.

-   **Dự đoán từ text:**

    ```bash
    python predict.py --text "Nội dung bài báo cần dự đoán..."
    ```

-   **Dự đoán từ file:**

    ```bash
    python predict.py --file path/to/your/article.txt
    ```

-   **Trực quan hóa Attention:** Thêm cờ `--visualize`.

    ```bash
    python predict.py --file path/to/your/article.txt --visualize
    ```

-   **Lưu biểu đồ Attention:**

    ```bash
    python predict.py --file path/to/your/article.txt --visualize --save_plot attention_plot.png
    ```

-   **Sử dụng model/vocab cụ thể:**
    ```bash
    python predict.py --text "..." --model path/to/model.pt [--raw --vocab path/to/vocab.txt]
    ```

Script sẽ sử dụng dữ liệu đã xử lý và model `saved_models/best_model.pt` mặc định nếu không có tùy chọn `--raw`, `--model`, `--vocab`.

### 5. Thử nghiệm siêu tham số (`main.py`)

Script này dùng để chạy thử nghiệm với nhiều cấu hình siêu tham số được định nghĩa trong `CONFIGURATIONS`. Script này **luôn sử dụng dữ liệu đã tiền xử lý** trong `processed_data/`. **Yêu cầu chạy `preprocess_data.py` trước.**

-   **Chạy tất cả các cấu hình (mỗi cấu hình 3 lần):**

    ```bash
    python main.py --all
    ```

-   **Chạy các cấu hình cụ thể:**

    ```bash
    python main.py --configs config_1 config_3 config_5
    ```

-   **Thay đổi số lần chạy cho mỗi cấu hình:**
    ```bash
    python main.py --all --runs 5
    ```

Kết quả chi tiết của từng lần chạy (model, history, plot) được lưu trong `experiments/config_X/run_Y/`. Kết quả tổng hợp (trung bình, độ lệch chuẩn) và biểu đồ so sánh được lưu trong `results/`. Toàn bộ quá trình được log lên Weights & Biases.

## Cấu hình

Các tham số mặc định (kích thước embedding, hidden dim, learning rate, số epochs cho `train.py`, đường dẫn...) được định nghĩa trong file `config.py`. Bạn có thể chỉnh sửa file này nếu cần thay đổi các giá trị mặc định.

Các cấu hình siêu tham số cho `main.py` được định nghĩa trực tiếp trong biến `CONFIGURATIONS` của file `main.py`.

## Kết quả và Phân tích

Kết quả tổng hợp và phân tích chi tiết của quá trình thử nghiệm siêu tham số (chạy bằng `main.py`) được trình bày trong file [report.md](report.md).

Bạn cũng có thể xem log chi tiết và biểu đồ tương tác trên Weights & Biases tại:
[https://wandb.ai/dohoangvunt2005/vietnamese-news-classification](https://wandb.ai/dohoangvunt2005/vietnamese-news-classification)
