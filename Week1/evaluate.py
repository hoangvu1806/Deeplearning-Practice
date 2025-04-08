import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import NeuralNetwork
import random
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model():
    # Kiểm tra thư mục checkpoint
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
        print("Đã tạo thư mục checkpoint")

    # Tải dữ liệu test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./dataset', train=False, download=False, transform=transform)

    # Chuyển đổi dữ liệu thành numpy arrays
    test_data = []
    test_labels = []
    for images, labels in test_dataset:
        test_data.append(images.numpy().flatten())
        test_labels.append(labels)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    print(f"Đã tải {len(test_data)} ảnh test")

    # Tải mô hình từ checkpoint
    checkpoint_file = 'checkpoint/best_model.npy'
    if os.path.exists(checkpoint_file):
        model = NeuralNetwork.load_checkpoint(checkpoint_file)
        print("Đã tải mô hình thành công")
    else:
        print(f"Không tìm thấy file checkpoint {checkpoint_file}")
        print("Vui lòng huấn luyện mô hình và lưu checkpoint trước")
        return

    # Lấy 20 ảnh ngẫu nhiên
    num_samples = 20
    indices = random.sample(range(len(test_data)), num_samples)
    sample_images = test_data[indices]
    sample_labels = test_labels[indices]
    
    # Dự đoán
    predictions = model.predict(sample_images)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(4, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if predictions[i] == sample_labels[i] else 'red'
        plt.title(f'Pred: {predictions[i]}\nTrue: {sample_labels[i]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_samples.png')
    plt.close()
    print("Đã lưu ảnh mẫu vào evaluation_samples.png")
    
    # Tính độ chính xác
    accuracy = np.mean(predictions == sample_labels)
    print(f'Độ chính xác trên mẫu: {accuracy:.2%}')
    
    # Hiển thị ma trận nhầm lẫn
    cm = confusion_matrix(sample_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn trên mẫu')
    plt.savefig('confusion_matrix_samples.png')
    plt.close()
    print("Đã lưu ma trận nhầm lẫn mẫu vào confusion_matrix_samples.png")

    # Đánh giá trên toàn bộ tập test
    all_predictions = model.predict(test_data)
    accuracy = np.mean(all_predictions == test_labels)
    print(f'Độ chính xác trên toàn bộ tập test: {accuracy:.2%}')
    
    # Hiển thị ma trận nhầm lẫn trên toàn bộ tập test
    cm = confusion_matrix(test_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn trên toàn bộ tập test')
    plt.savefig('confusion_matrix_full.png')
    plt.close()
    print("Đã lưu ma trận nhầm lẫn đầy đủ vào confusion_matrix_full.png")

if __name__ == "__main__":
    evaluate_model() 