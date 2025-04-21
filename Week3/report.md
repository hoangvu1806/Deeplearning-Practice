# Báo cáo phân tích California Housing Regression

## Giới thiệu

Báo cáo này phân tích kết quả từ việc huấn luyện các mô hình neural network khác nhau trên bộ dữ liệu California Housing. Mục tiêu là xác định cấu trúc mạng và các siêu tham số tối ưu cho bài toán dự đoán giá nhà.

## Thông tin dự án

-   **Weights & Biases Dashboard**: [https://wandb.ai/dohoangvunt2005/california_housing_regression](https://wandb.ai/dohoangvunt2005/california_housing_regression)
-   Tất cả logs, biểu đồ và kết quả chi tiết có thể được xem trực tiếp tại liên kết trên

## Phương pháp

### Dữ liệu

Bộ dữ liệu California Housing chứa thông tin về giá nhà ở California với các đặc trưng như:

-   Độ tuổi trung bình
-   Số phòng trung bình
-   Thu nhập trung bình
-   Vị trí địa lý
-   Các chỉ số nhân khẩu học khác

### Mô hình

Chúng tôi đã thử nghiệm nhiều cấu trúc mô hình khác nhau:

1. **Small Network**: Mạng neural đơn giản với 2 lớp ẩn [32, 16]
2. **Medium Network**: Mạng neural với 3 lớp ẩn [64, 32, 16]
3. **Large Network**: Mạng neural phức tạp với 3 lớp ẩn [128, 64, 32]
4. **Learning Rate Variations**: Mô hình với các learning rate khác nhau (0.001, 0.01)

### Quy trình huấn luyện

-   Mỗi cấu hình được chạy nhiều lần để đảm bảo kết quả đáng tin cậy
-   Sử dụng 5-fold cross-validation để đánh giá hiệu suất
-   Theo dõi các metric: RMSE, R², MSE, MAE trong quá trình huấn luyện và kiểm thử
-   Sử dụng Weights & Biases để theo dõi và ghi lại các thí nghiệm

## Kết quả

### So sánh đường cong Loss

Phân tích đường cong loss trong quá trình huấn luyện cho thấy:

-   Small Network hội tụ nhanh hơn nhưng có xu hướng underfitting
-   Medium Network cân bằng tốt giữa tốc độ hội tụ và hiệu suất
-   Large Network có thể đạt được loss thấp hơn nhưng mất nhiều thời gian hơn để hội tụ
-   Learning rate thấp (0.001) tạo ra đường cong loss mượt hơn nhưng hội tụ chậm
-   Learning rate cao (0.01) có thể gây ra dao động trong quá trình huấn luyện

### So sánh Metric

#### RMSE (Root Mean Squared Error)

-   Medium Network đạt được RMSE thấp nhất trên tập test
-   Small Network có RMSE cao hơn, chỉ ra rằng nó có thể không đủ phức tạp
-   Large Network có xu hướng overfitting, dẫn đến hiệu suất không tốt trên tập test

#### R² (R-squared)

-   Medium Network đạt giá trị R² cao nhất, cho thấy khả năng giải thích cao nhất
-   Learning rate phù hợp (0.001) cho kết quả R² cao hơn so với learning rate thấp hoặc cao

#### MSE và MAE

-   Xu hướng tương tự như RMSE, trong đó cấu trúc trung bình đạt được hiệu suất tốt nhất

### Ảnh hưởng của Siêu tham số

#### Learning Rate

-   0.001 dường như là learning rate tối ưu cho hầu hết các cấu trúc mạng
-   Learning rate thấp hơn (0.0001) quá chậm để hội tụ
-   Learning rate cao hơn (0.01) gây ra dao động và không ổn định

#### Batch Size

-   Batch size trung bình (32, 64) mang lại hiệu suất tốt nhất
-   Batch size nhỏ (16) cho phép cập nhật thường xuyên hơn nhưng tạo ra nhiều nhiễu
-   Batch size lớn (128) ổn định hơn nhưng hội tụ chậm hơn

#### Dropout Rate

-   Dropout rate 0.2 cho thấy cân bằng tốt giữa regularization và duy trì khả năng dự đoán
-   Không có dropout (0.0) dẫn đến overfitting trên mạng lớn
-   Dropout quá cao (>0.5) làm giảm hiệu suất của tất cả các mô hình

#### Kiến trúc mạng

-   Mạng 3 lớp với kích thước [64, 32, 16] đạt được hiệu suất tốt nhất
-   Mạng sâu hơn không nhất thiết mang lại cải thiện đáng kể
-   Sự cân bằng giữa sự phức tạp của mô hình và kích thước dữ liệu là quan trọng

## Phân tích và Kết luận

### Cấu hình tốt nhất

Dựa trên phân tích của chúng tôi, cấu hình tối ưu cho bài toán California Housing Regression là:

-   Kiến trúc: Medium Network (3 lớp ẩn [64, 32, 16])
-   Learning rate: 0.001
-   Batch size: 64
-   Dropout rate: 0.2
-   Epochs: 100

### Bài học rút ra

1. **Hiệu ứng Goldilocks**: Mô hình "vừa đủ" hoạt động tốt hơn so với mô hình quá đơn giản (underfitting) hoặc quá phức tạp (overfitting)
2. **Tầm quan trọng của learning rate**: Việc chọn learning rate phù hợp có tác động lớn đến hiệu suất và tốc độ hội tụ
3. **Regularization**: Dropout với tỷ lệ phù hợp cải thiện đáng kể khả năng tổng quát hóa
4. **Variability trong kết quả**: Chạy mô hình nhiều lần cho thấy độ ổn định của các cấu hình khác nhau

### Hướng phát triển trong tương lai

1. **Grid Search**: Thực hiện tìm kiếm lưới chi tiết hơn cho siêu tham số tối ưu
2. **Transfer Learning**: Khám phá các kiến trúc pre-trained cho các đặc trưng địa lý
3. **Feature Engineering**: Tạo các đặc trưng mới từ dữ liệu hiện có để cải thiện hiệu suất dự đoán
4. **Ensemble Methods**: Kết hợp nhiều mô hình để cải thiện độ chính xác và độ tin cậy của dự đoán

## Tóm tắt

Phân tích này đã xác định các cấu hình tối ưu cho bài toán dự đoán giá nhà California. Medium Network với cấu trúc [64, 32, 16] và learning rate 0.001 đạt được hiệu suất tốt nhất về cả RMSE và R². Kết quả này có thể áp dụng cho các bài toán hồi quy tương tự trong lĩnh vực bất động sản.
