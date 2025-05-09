# Phân loại ảnh CIFAR-10 với CNN và ResNet-20

Dự án này thực hiện phân loại ảnh CIFAR-10 sử dụng hai kiến trúc mạng: CNN cơ bản và ResNet-20. Mục tiêu là thử nghiệm và so sánh hiệu suất của các mô hình với nhiều cấu hình siêu tham số khác nhau.

## Tổng quan

-   **Tập dữ liệu**: CIFAR-10 (10 lớp, 60.000 ảnh kích thước 32x32)
-   **Kiến trúc**: CNN cơ bản và ResNet-20
-   **Theo dõi thử nghiệm**: Weights & Biases (wandb)
-   **Demo**: Ứng dụng Streamlit cho phép phân loại ảnh với các mô hình đã huấn luyện

## Cấu trúc dự án

```
Week5/
├── app.py              # Ứng dụng demo Streamlit
├── configs/
│   └── config.json     # Cấu hình siêu tham số cho các mô hình
├── dataset/            # Thư mục chứa dữ liệu CIFAR-10
├── models/             # Thư mục lưu trữ các mô hình đã huấn luyện
├── models.py           # Định nghĩa kiến trúc các mô hình CNN và ResNet-20
├── plots/              # Biểu đồ và trực quan hóa
│   └── feature_maps/   # Trực quan hóa feature maps
├── requirements.txt    # Các thư viện cần thiết
├── results/            # Kết quả thử nghiệm (JSON)
├── train.py            # Mã nguồn huấn luyện và đánh giá mô hình
├── main.py             # Mã nguồn chính để chạy thử nghiệm
└── utils.py            # Các hàm tiện ích
```

## Cài đặt và Yêu cầu

```bash
pip install -r requirements.txt
```

Các thư viện chính yêu cầu:

-   PyTorch
-   torchvision
-   NumPy
-   Matplotlib
-   tqdm
-   Weights & Biases (wandb)
-   Streamlit

## Cách sử dụng

### 1. Cấu hình siêu tham số

Các cấu hình siêu tham số được định nghĩa trong file `configs/config.json`. Bạn có thể điều chỉnh các tham số sau:

**CNN**:

-   `num_conv_layers`: Số lớp tích chập
-   `base_filters`: Số filter cơ sở (sẽ tăng gấp đôi qua mỗi lớp)
-   `learning_rate`: Tốc độ học
-   `batch_size`: Kích thước batch
-   `optimizer`: Thuật toán tối ưu (adam, sgd)
-   `epochs`: Số epoch huấn luyện

**ResNet**:

-   `base_filters`: Số filter cơ sở
-   `learning_rate`: Tốc độ học
-   `batch_size`: Kích thước batch
-   `optimizer`: Thuật toán tối ưu
-   `epochs`: Số epoch huấn luyện

### 2. Huấn luyện mô hình

```bash
python main.py
```

Quá trình huấn luyện sẽ:

-   Tự động tải và tiền xử lý dữ liệu CIFAR-10
-   Huấn luyện cả CNN và ResNet-20 với ba cấu hình khác nhau
-   Lưu các mô hình tốt nhất vào thư mục `models/`
-   Lưu kết quả vào thư mục `results/`
-   Tạo biểu đồ hiệu suất trong thư mục `plots/`
-   Theo dõi quá trình huấn luyện với Weights & Biases

### 3. Chạy ứng dụng demo

```bash
streamlit run app.py
```

Ứng dụng demo cho phép:

-   Chọn loại mô hình (CNN hoặc ResNet-20)
-   Chọn một trong ba cấu hình đã huấn luyện
-   Tải lên ảnh để phân loại
-   Hiển thị kết quả phân loại với xác suất dự đoán

## Mô hình và Kiến trúc

### CNN cơ bản

Mô hình CNN cơ bản bao gồm:

-   Nhiều khối tích chập với bố cục: Conv2d → BatchNorm2d → ReLU → MaxPool2d
-   Số filter tăng gấp đôi qua mỗi lớp tích chập (ví dụ: 32 → 64 → 128 → ...)
-   Lớp fully connected ở cuối để ánh xạ vào 10 lớp

### ResNet-20

Cài đặt của ResNet-20 được thiết kế riêng cho CIFAR-10:

-   20 lớp tích chập tổng cộng
-   3 khối chính, mỗi khối có 3 basic block (mỗi basic block gồm 2 lớp tích chập)
-   Kết nối tắt (shortcut connections) giúp giải quyết vấn đề tiêu biến gradient
-   Global Average Pooling thay cho Fully Connected layers

## Kết quả và Phân tích

Kết quả chi tiết của các thử nghiệm được lưu trong thư mục `results/` dưới dạng file JSON:

-   `cnn_results.json`: Kết quả của các mô hình CNN
-   `resnet_results.json`: Kết quả của các mô hình ResNet
-   `comparison.json`: So sánh tổng hợp giữa hai kiến trúc

Các biểu đồ hiệu suất được lưu trong thư mục `plots/`:

-   Biểu đồ độ chính xác (accuracy)
-   Biểu đồ mất mát (loss)
-   Trực quan hóa feature maps

## Tác giả

Email: dohoangvuk16@siu.edu.vn
