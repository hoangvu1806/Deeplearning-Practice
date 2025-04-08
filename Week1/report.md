# Báo cáo bài tập tuần 1: Nhận dạng Chữ số Viết tay MNIST

Tác giả: Đỗ Hoàng Vũ

## 1. Tổng quan

### 1.1. Giới thiệu

Bài tập này tập trung vào việc xây dựng và huấn luyện một mạng neural đơn giản để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST. Mục tiêu chính là tìm hiểu và áp dụng các khái niệm cơ bản của deep learning như forward propagation, backward propagation, và tối ưu hóa gradient descent.

### 1.2. Mục tiêu

-   Xây dựng mạng neural từ đầu bằng NumPy
-   Thử nghiệm với các hàm kích hoạt và siêu tham số khác nhau
-   Đạt độ chính xác tốt trên tập test
-   Hiểu rõ về cách hoạt động của mạng neural

## 2. Kiến trúc Mô hình

### 2.1. Tổng quan Kiến trúc

-   **Input Layer**: 784 neurons (28x28 pixels)
-   **Hidden Layer**: Số lượng neurons có thể điều chỉnh (16-128)
-   **Output Layer**: 10 neurons (0-9)

### 2.2. Hàm Kích hoạt

-   **Hidden Layer**:
    -   ReLU: f(x) = max(0, x)
    -   Sigmoid: f(x) = 1/(1 + e^(-x))
    -   Tanh: f(x) = tanh(x)
-   **Output Layer**:
    -   Softmax: f(x_i) = e^(x_i)/Σe^(x_j)

### 2.3. Khởi tạo Trọng số

-   W1: np.random.randn(input_size, hidden_size) \* np.sqrt(2.0/input_size)
-   W2: np.random.randn(hidden_size, output_size) \* np.sqrt(2.0/hidden_size)
-   Bias: Khởi tạo bằng 0

## 3. Quá trình Huấn luyện

### 3.1. Dữ liệu

-   **Tập huấn luyện**: 60,000 ảnh
-   **Tập kiểm tra**: 10,000 ảnh
-   **Tiền xử lý**:
    -   Chuẩn hóa dữ liệu với mean=0.1307, std=0.3081
    -   Chuyển đổi nhãn sang one-hot encoding

### 3.2. Chiến lược Huấn luyện

-   **Mini-batch Gradient Descent**
-   **Cross-entropy Loss**
-   **Thử nghiệm các siêu tham số**:
    ```python
    hyperparameters = [
        {"batch_size": 32, "learning_rate": 0.1, "hidden_size": 16, "activation": "relu"},
        {"batch_size": 16, "learning_rate": 0.01, "hidden_size": 64, "activation": "sigmoid"},
        {"batch_size": 64, "learning_rate": 0.05, "hidden_size": 32, "activation": "tanh"},
        {"batch_size": 128, "learning_rate": 0.001, "hidden_size": 128, "activation": "relu"},
        {"batch_size": 32, "learning_rate": 0.01, "hidden_size": 64, "activation": "sigmoid"}
    ]
    ```

## 4. Kết quả và Phân tích

### 4.1. So sánh Hiệu suất

| Cấu hình | Batch Size | Learning Rate | Hidden Size | Activation | Độ chính xác | Độ lệch chuẩn |
| -------- | ---------- | ------------- | ----------- | ---------- | ------------ | ------------- |
| 1        | 32         | 0.1           | 16          | ReLU       | 94.83%       | 0.29%         |
| 2        | 16         | 0.01          | 64          | Sigmoid    | 95.14%       | 0.13%         |
| 3        | 64         | 0.05          | 32          | Tanh       | 95.76%       | 0.07%         |
| 4        | 128        | 0.001         | 128         | ReLU       | 90.47%       | 0.17%         |
| 5        | 32         | 0.01          | 64          | Sigmoid    | 93.97%       | 0.13%         |

Nhận xét:

-   Cấu hình tốt nhất là #3 với độ chính xác 95.76% và độ lệch chuẩn thấp nhất (0.07%)
-   Cấu hình kém nhất là #4 với độ chính xác 90.47%, có thể do learning rate quá nhỏ (0.001)
-   Hàm kích hoạt Tanh cho kết quả tốt nhất trong thử nghiệm này

### 4.2. Phân tích Ma trận Nhầm lẫn

-   Xem file `confusion_matrix_full.png`
-   Các chữ số dễ nhầm lẫn nhất:
    -   4 và 9 (do cấu trúc tương tự phần trên)
    -   3 và 8 (do đều có các đường cong)
    -   5 và 3 (do đều có phần cong ở giữa)

### 4.3. Ví dụ Dự đoán

-   Xem file `evaluation_samples.png` để thấy các ví dụ cụ thể
-   Màu xanh: dự đoán đúng
-   Màu đỏ: dự đoán sai

## 5. Kết luận và Cải tiến

### 5.1. Kết luận

-   Mô hình đạt hiệu suất tốt (95.76%) với kiến trúc đơn giản
-   Tanh cho kết quả tốt nhất trong thử nghiệm này
-   Batch size trung bình (64) và learning rate vừa phải (0.05) cho kết quả tốt nhất
-   Độ lệch chuẩn thấp (0.07%-0.29%) cho thấy mô hình ổn định

### 5.2. Cải tiến trong Tương lai

1. **Kiến trúc Mô hình**:

    - Thêm nhiều lớp ẩn
    - Thử nghiệm với các kiến trúc phức tạp hơn (CNN)
    - Tăng số lượng neurons trong lớp ẩn

2. **Kỹ thuật Tối ưu hóa**:

    - Thêm regularization (L1/L2) để giảm overfitting
    - Sử dụng dropout
    - Thử nghiệm các optimizer khác (Adam, RMSprop)
    - Thử nghiệm learning rate scheduling

3. **Tiền xử lý Dữ liệu**:
    - Augmentation dữ liệu (xoay, co giãn, thêm nhiễu)
    - Thử nghiệm các phương pháp chuẩn hóa khác
    - Cân bằng dữ liệu nếu cần

## 6. Tài liệu Tham khảo

1. LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit database. http://yann.lecun.com/exdb/mnist/
2. Nielsen, M. (2015). Neural Networks and Deep Learning. http://neuralnetworksanddeeplearning.com/
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
