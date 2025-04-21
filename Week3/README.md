# Dự án Hồi Quy Giá Nhà California

## Tổng quan

Dự án này sử dụng mạng neural để dự đoán giá nhà ở California dựa trên bộ dữ liệu California Housing. Mục tiêu chính là phân tích và so sánh hiệu suất của các cấu trúc mạng neural và tham số khác nhau trên bài toán hồi quy này.

## Thông tin dự án

-   **Dashboard**: [Weights & Biases Project](https://wandb.ai/dohoangvunt2005/california_housing_regression)
-   Truy cập vào link trên để xem trực quan kết quả các thí nghiệm và so sánh các mô hình.

## Mô tả Dự án

Dự án thực hiện huấn luyện và đánh giá các mô hình mạng neural với năm cấu hình cụ thể:

-   Small Network: Mạng neural đơn giản
-   Medium Network: Mạng neural kích thước trung bình
-   Large Network: Mạng neural phức tạp
-   Learning Rate Small: Tốc độ học thấp
-   Learning Rate Large: Tốc độ học cao

## Cấu trúc mã nguồn

-   `california_housing_regression.py`: File chính để huấn luyện các mô hình
-   `visualize_results.py`: Script phân tích và trực quan hóa kết quả từ Weights & Biases
-   `results/`: Thư mục chứa các biểu đồ và kết quả phân tích (tự động tạo)

## Các cấu hình chính xác được thử nghiệm

Dự án đánh giá năm cấu hình mạng neural sau:

1. **Small Network**:

    - Hidden layers: [32, 16]
    - Batch size: 64
    - Learning rate: 0.001
    - Dropout rate: 0.1
    - Epochs: 100

2. **Medium Network**:

    - Hidden layers: [64, 32, 16]
    - Batch size: 64
    - Learning rate: 0.001
    - Dropout rate: 0.2
    - Epochs: 100

3. **Large Network**:

    - Hidden layers: [128, 64, 32]
    - Batch size: 32
    - Learning rate: 0.001
    - Dropout rate: 0.3
    - Epochs: 100

4. **Learning Rate Small**:

    - Hidden layers: [64, 32]
    - Batch size: 64
    - Learning rate: 0.0001
    - Dropout rate: 0.2
    - Epochs: 100

5. **Learning Rate Large**:
    - Hidden layers: [64, 32]
    - Batch size: 64
    - Learning rate: 0.01
    - Dropout rate: 0.2
    - Epochs: 100

## Các metric đánh giá hiệu suất

-   **RMSE** (Root Mean Squared Error): Đo lường sai số trung bình
-   **R²** (R-squared): Chỉ số xác định mức độ phù hợp của mô hình
-   **MSE** (Mean Squared Error): Đo lường sai số bình phương trung bình
-   **MAE** (Mean Absolute Error): Đo lường sai số tuyệt đối trung bình

## Công cụ theo dõi và phân tích

Dự án sử dụng **Weights & Biases (wandb)** để:

-   Ghi lại quá trình huấn luyện
-   Theo dõi và so sánh các metric
-   Trực quan hóa kết quả

## Cách sử dụng

### Yêu cầu

```
pytorch
numpy
pandas
matplotlib
seaborn
scikit-learn
wandb
tqdm
```

### Cài đặt

```bash
pip install -r requirements.txt
```

### Huấn luyện mô hình

```bash
python california_housing_regression.py
```

### Phân tích kết quả

```bash
python visualize_results.py
```

## Phân tích kết quả

Dự án tạo ra các phân tích sau:

1. **Đường cong Loss**: So sánh quá trình hội tụ của các mô hình
2. **So sánh metrics**: Đánh giá RMSE, R², MSE, MAE giữa các cấu hình
3. **Phân tích siêu tham số**: Ảnh hưởng của learning rate, batch size, và dropout
4. **Bảng thống kê**: Tổng hợp hiệu suất của các cấu hình khác nhau

Chi tiết về kết quả phân tích có thể xem trong [report.md](report.md).

## Kết luận

Dựa trên kết quả phân tích, Medium Network với learning rate 0.001 và batch size 64 cho kết quả tốt nhất, cân bằng giữa độ phức tạp và khả năng dự đoán.
