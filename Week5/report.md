# Báo cáo Kết quả Thử nghiệm Siêu tham số: Phân loại ảnh CIFAR-10 với CNN và ResNet-20

## 1. Giới thiệu

Báo cáo này trình bày chi tiết về việc thực hiện và đánh giá hai kiến trúc mạng nơ-ron tích chập khác nhau cho bài toán phân loại ảnh CIFAR-10: mạng CNN cơ bản và kiến trúc ResNet-20. Mục tiêu chính là so sánh hiệu suất của các mô hình này với các cấu hình siêu tham số khác nhau, đồng thời đánh giá ảnh hưởng của các shortcut connections trong kiến trúc ResNet đối với khả năng phân loại ảnh.

## 2. Tập dữ liệu và Tiền xử lý

### Tập dữ liệu CIFAR-10

CIFAR-10 là một tập dữ liệu chuẩn trong thị giác máy tính, bao gồm:

-   60.000 ảnh màu RGB kích thước 32x32 pixel
-   10 lớp đối tượng: máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu thủy, xe tải
-   50.000 ảnh huấn luyện và 10.000 ảnh kiểm tra

### Tiền xử lý

-   **Chuẩn hóa**: Các ảnh được chuẩn hóa với giá trị trung bình (0.4914, 0.4822, 0.4465) và độ lệch chuẩn (0.2470, 0.2435, 0.2616) trên ba kênh màu
-   **Tăng cường dữ liệu**: Áp dụng các kỹ thuật như lật ngang ngẫu nhiên và cắt ngẫu nhiên để tăng cường tập huấn luyện

## 3. Kiến trúc mô hình

### 3.1 CNN cơ bản

Mô hình CNN được thiết kế với cấu trúc đơn giản nhưng hiệu quả:

-   **Đặc điểm chính**:

    -   Nhiều khối tích chập, số lượng khác nhau dựa trên cấu hình (3-5 lớp)
    -   Mỗi khối bao gồm: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    -   Số lượng filter tăng gấp đôi qua mỗi lớp (ví dụ: 32 → 64 → 128)
    -   Fully Connected layer ở cuối mạng

-   **Các cấu hình thử nghiệm**:
    1. **Config 1**: 3 lớp tích chập, bắt đầu với 32 filter, learning rate 0.001 (Adam)
    2. **Config 2**: 4 lớp tích chập, bắt đầu với 64 filter, learning rate 0.0005 (SGD)
    3. **Config 3**: 5 lớp tích chập, bắt đầu với 32 filter, learning rate 0.0001 (Adam)

### 3.2 ResNet-20

Kiến trúc ResNet-20 được thiết kế đặc biệt cho tập dữ liệu CIFAR:

-   **Đặc điểm chính**:

    -   20 lớp tích chập, được tổ chức thành 3 khối chính
    -   Mỗi khối chính chứa 3 basic block, mỗi basic block có 2 lớp tích chập
    -   Kết nối tắt (residual/shortcut connections) để giải quyết vấn đề tiêu biến gradient
    -   Global Average Pooling thay cho Fully Connected layers
    -   Kích thước đặc trưng giảm dần: 32×32 → 16×16 → 8×8

-   **Các cấu hình thử nghiệm**:
    1. **Config 1**: 16 filter cơ sở, learning rate 0.1 (SGD)
    2. **Config 2**: 32 filter cơ sở, learning rate 0.01 (SGD)
    3. **Config 3**: 16 filter cơ sở, learning rate 0.001 (Adam)

## 4. Phương pháp Huấn luyện

### Chiến lược huấn luyện

-   **Loss function**: Cross-Entropy
-   **Tối ưu hóa**: Thử nghiệm với Adam và SGD
-   **Epochs**: 30 epochs cho mỗi cấu hình
-   **Batch size**: Thử nghiệm với 64, 128, và 256
-   **Model selection**: Lưu mô hình có độ chính xác validation cao nhất

### Quản lý thử nghiệm

-   Sử dụng Weights & Biases (wandb) để theo dõi và ghi lại các thử nghiệm
-   Toàn bộ logs quá trình huấn luyện được lưu trữ và có thể xem tại: [https://wandb.ai/dohoangvunt2005/cifar10-classification](https://wandb.ai/dohoangvunt2005/cifar10-classification)
-   Lưu lại thông số mô hình, đường cong học tập và kết quả đánh giá
-   Trực quan hóa feature maps để hiểu rõ hơn về cách các mô hình hoạt động

## 5. Kết quả và Phân tích

### 5.1 Hiệu suất tổng thể

Từ kết quả trong file comparison.json, chúng ta có số liệu tổng thể về CNN và ResNet-20:

| Mô hình | Độ chính xác trung bình (%) | Độ lệch chuẩn (%) | Loss trung bình | Độ lệch chuẩn loss |
| ------- | --------------------------- | ----------------- | --------------- | ------------------ |
| CNN     | 82.56                       | 0.15              | 0.522           | 0.005              |
| ResNet  | 85.93                       | 1.93              | 0.443           | 0.093              |

### 5.2 So sánh các cấu hình CNN

| Cấu hình | Siêu tham số chính           | Độ chính xác kiểm tra (%) | Thời gian huấn luyện (s) |
| -------- | ---------------------------- | ------------------------- | ------------------------ |
| Config 1 | 3 lớp, 32 filter cơ sở, Adam | 82.73                     | 462.49                   |
| Config 2 | 4 lớp, 64 filter cơ sở, SGD  | 82.37                     | 1042.59                  |
| Config 3 | 5 lớp, 32 filter cơ sở, Adam | 82.57                     | 701.73                   |

Nhận xét:

-   Cấu hình 1 đạt độ chính xác cao nhất trong các mô hình CNN
-   Tăng độ sâu không cải thiện đáng kể hiệu suất, có thể do vấn đề tiêu biến gradient
-   Thời gian huấn luyện tăng đáng kể khi tăng số lớp và số filter

### 5.3 So sánh các cấu hình ResNet

| Cấu hình | Siêu tham số chính              | Độ chính xác kiểm tra (%) | Thời gian huấn luyện (s) |
| -------- | ------------------------------- | ------------------------- | ------------------------ |
| Config 1 | 16 filter cơ sở, lr=0.1, SGD    | 87.42                     | 580.61                   |
| Config 2 | 32 filter cơ sở, lr=0.01, SGD   | 86.68                     | 699.04                   |
| Config 3 | 16 filter cơ sở, lr=0.001, Adam | 83.68                     | 587.02                   |

Nhận xét:

-   Cấu hình 1 đạt hiệu suất tốt nhất, với SGD và learning rate cao
-   Tăng số filter cơ sở không cải thiện hiệu suất nhưng làm tăng thời gian huấn luyện
-   Adam có vẻ kém hiệu quả hơn SGD cho ResNet với tập dữ liệu này

### 5.4 Đường cong học tập

Phân tích đường cong học tập cho thấy:

-   **CNN**:

    -   Hội tụ ổn định nhưng chậm
    -   Độ chính xác validation đạt khoảng 80% sau 30 epochs
    -   Có hiện tượng overfitting nhẹ trong các epochs cuối

-   **ResNet**:
    -   Hội tụ nhanh hơn, đặc biệt với config 1 (SGD, lr=0.1)
    -   Đạt độ chính xác validation trên 85%
    -   Thể hiện khả năng học tốt hơn với số lượng tham số tương đương CNN

### 5.5 Trực quan hóa feature maps

Trực quan hóa feature maps cho thấy:

-   **CNN**:

    -   Các lớp đầu phát hiện cạnh và đường viền cơ bản
    -   Các lớp sau phát hiện các đặc trưng phức tạp hơn như hình dạng
    -   Feature maps có xu hướng mờ dần ở các lớp sâu hơn

-   **ResNet**:
    -   Các feature maps chi tiết và rõ ràng hơn ở các lớp sâu
    -   Kết nối tắt giúp bảo toàn thông tin về đặc trưng từ các lớp nông
    -   Phát hiện được nhiều đặc trưng phức tạp hơn ở các lớp cuối

## 6. Thảo luận

### 6.1 So sánh CNN và ResNet

-   **Hiệu suất**: ResNet-20 vượt trội hơn so với CNN cơ bản, đạt độ chính xác cao hơn khoảng 3.5%
-   **Độ ổn định**: ResNet có sự biến thiên lớn hơn giữa các cấu hình (độ lệch chuẩn 1.93% so với 0.15%)
-   **Thời gian huấn luyện**: Thời gian huấn luyện trung bình của ResNet (622.22s) thấp hơn CNN (735.60s) khi xét hiệu suất tương đương

### 6.2 Tầm quan trọng của kết nối tắt

Kết nối tắt trong ResNet đã chứng minh hiệu quả vượt trội:

-   Giúp mạng học sâu hơn mà không gặp vấn đề tiêu biến gradient
-   Cho phép thông tin từ các lớp nông lan truyền trực tiếp đến các lớp sâu
-   Cải thiện đáng kể khả năng phân loại, đặc biệt đối với các lớp khó phân biệt

### 6.3 Tối ưu hóa và Siêu tham số

Phân tích kết quả cho thấy:

-   SGD với learning rate cao hoạt động tốt hơn Adam cho ResNet
-   Cấu trúc đơn giản với học chậm (Config 1 của CNN) có thể hiệu quả hơn các mô hình phức tạp
-   Batch size lớn (256) không nhất thiết cải thiện hiệu suất so với batch size trung bình (128)

## 7. Kết luận

Thử nghiệm với CIFAR-10 cho thấy:

1. ResNet-20 vượt trội hơn CNN cơ bản trong bài toán phân loại ảnh CIFAR-10
2. Kết nối tắt là yếu tố quan trọng giúp các mạng sâu đạt hiệu suất cao
3. Việc chọn bộ tối ưu và cấu hình siêu tham số phù hợp có ảnh hưởng lớn đến hiệu suất
4. Mô hình đơn giản được huấn luyện tốt có thể hiệu quả hơn mô hình phức tạp huấn luyện không đủ

## 8. Hướng phát triển

Các hướng nghiên cứu và phát triển tiếp theo:

1. Thử nghiệm với các kiến trúc hiện đại hơn như DenseNet, EfficientNet
2. Áp dụng các kỹ thuật tăng cường dữ liệu nâng cao
3. Điều chỉnh learning rate tự động trong quá trình huấn luyện
4. Thử nghiệm các phương pháp regularization như dropout, weight decay
5. Phân tích độ chính xác theo từng lớp để xác định điểm yếu của mô hình
