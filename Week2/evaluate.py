import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from models import RNN, LSTM
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import re
from nltk.tokenize import word_tokenize
import nltk

# Đảm bảo nltk đã tải các gói cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_results():
    with open("results/results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return results

def plot_accuracy_comparison(results):
    model_types = [r["config"]["model_type"] for r in results]
    mean_accs = [r["mean_acc"] for r in results]
    std_accs = [r["std_acc"] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(mean_accs)), mean_accs, yerr=std_accs, 
                  capsize=10, alpha=0.7, color=['blue' if m == 'LSTM' else 'green' for m in model_types])
    
    # Thêm nhãn cho mỗi cột
    config_labels = []
    for r in results:
        config = r["config"]
        label = f"{config['model_type']}\nHL: {config['hidden_size']}\nL: {config['num_layers']}\nBS: {config['batch_size']}\nOpt: {config['optimizer_type']}"
        config_labels.append(label)
    
    ax.set_xticks(range(len(mean_accs)))
    ax.set_xticklabels(config_labels, rotation=0)
    
    # Thêm các giá trị cụ thể
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_accs[i]:.4f}', ha='center', va='bottom')
    
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Độ chính xác trung bình')
    ax.set_title('So sánh độ chính xác giữa các cấu hình mô hình')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300)
    plt.close()

def plot_model_type_comparison(results):
    # Tách kết quả theo loại mô hình
    lstm_results = [r for r in results if r["config"]["model_type"] == "LSTM"]
    rnn_results = [r for r in results if r["config"]["model_type"] == "RNN"]
    
    lstm_accs = [r["mean_acc"] for r in lstm_results]
    rnn_accs = [r["mean_acc"] for r in rnn_results]
    
    # Tính trung bình và độ lệch chuẩn cho từng loại mô hình
    lstm_mean = np.mean(lstm_accs)
    rnn_mean = np.mean(rnn_accs)
    lstm_std = np.std(lstm_accs)
    rnn_std = np.std(rnn_accs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['LSTM', 'RNN'], [lstm_mean, rnn_mean], yerr=[lstm_std, rnn_std], 
                  capsize=10, alpha=0.7, color=['blue', 'green'])
    
    # Thêm giá trị lên từng cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('Độ chính xác trung bình')
    ax.set_title('So sánh hiệu suất giữa LSTM và RNN')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/model_type_comparison.png', dpi=300)
    plt.close()

def load_model_and_evaluate(model_path, config, data_path="dataset/processed_data.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tải dữ liệu
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    texts = data["texts"]
    labels = data["labels"]
    vocab_dict = data["vocab"]
    pad_idx = vocab_dict["<pad>"]
    
    # Lấy dữ liệu kiểm thử
    TRAIN_SIZE = 25000
    TEST_SIZE = 5000
    test_texts = texts[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    test_labels = labels[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    
    # Tạo dataset và dataloader
    test_dataset = TensorDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Tạo mô hình tương ứng
    if config["model_type"] == "RNN":
        model = RNN(
            config["vocab_size"], config["embedding_dim"], config["hidden_size"], 
            config["num_layers"], config["output_size"], pad_idx
        )
    else:  # LSTM
        model = LSTM(
            config["vocab_size"], config["embedding_dim"], config["hidden_size"], 
            config["num_layers"], config["output_size"], pad_idx
        )
    
    # Tải trọng số đã lưu
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Đánh giá mô hình
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    print(f"Model: {config['model_type']}, Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tiêu cực', 'Tích cực'],
                yticklabels=['Tiêu cực', 'Tích cực'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(f'Ma trận nhầm lẫn - {config["model_type"]}')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{config["model_type"]}.png', dpi=300)
    plt.close()
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Tiêu cực', 'Tích cực'],
                                   output_dict=True)
    
    return test_acc, test_loss, report

def plot_hyperparameter_effects(results):
    # Phân tích ảnh hưởng của batch_size
    batch_sizes = [r["config"]["batch_size"] for r in results]
    accuracies = [r["mean_acc"] for r in results]
    model_types = [r["config"]["model_type"] for r in results]
    
    plt.figure(figsize=(10, 6))
    for mt in ['LSTM', 'RNN']:
        mt_indices = [i for i, m in enumerate(model_types) if m == mt]
        plt.scatter([batch_sizes[i] for i in mt_indices], 
                   [accuracies[i] for i in mt_indices],
                   label=mt, alpha=0.7, s=100)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Độ chính xác')
    plt.title('Ảnh hưởng của Batch Size đến độ chính xác')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/batch_size_effect.png', dpi=300)
    plt.close()
    
    # Phân tích ảnh hưởng của số lớp ẩn
    num_layers = [r["config"]["num_layers"] for r in results]
    
    plt.figure(figsize=(10, 6))
    for mt in ['LSTM', 'RNN']:
        mt_indices = [i for i, m in enumerate(model_types) if m == mt]
        plt.scatter([num_layers[i] for i in mt_indices], 
                   [accuracies[i] for i in mt_indices],
                   label=mt, alpha=0.7, s=100)
    
    plt.xlabel('Số lớp ẩn')
    plt.ylabel('Độ chính xác')
    plt.title('Ảnh hưởng của số lớp ẩn đến độ chính xác')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/num_layers_effect.png', dpi=300)
    plt.close()
    
    # Phân tích ảnh hưởng của optimizer
    optimizers = [r["config"]["optimizer_type"] for r in results]
    unique_optimizers = list(set(optimizers))
    
    opt_data = {opt: {'accs': [], 'model_types': []} for opt in unique_optimizers}
    
    for i, r in enumerate(results):
        opt = r["config"]["optimizer_type"]
        opt_data[opt]['accs'].append(r["mean_acc"])
        opt_data[opt]['model_types'].append(r["config"]["model_type"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(unique_optimizers))
    width = 0.35
    
    lstm_means = []
    rnn_means = []
    
    for i, opt in enumerate(unique_optimizers):
        lstm_accs = [acc for j, acc in enumerate(opt_data[opt]['accs']) 
                     if opt_data[opt]['model_types'][j] == 'LSTM']
        rnn_accs = [acc for j, acc in enumerate(opt_data[opt]['accs']) 
                    if opt_data[opt]['model_types'][j] == 'RNN']
        
        lstm_mean = np.mean(lstm_accs) if lstm_accs else 0
        rnn_mean = np.mean(rnn_accs) if rnn_accs else 0
        
        lstm_means.append(lstm_mean)
        rnn_means.append(rnn_mean)
    
    ax.bar(x_pos - width/2, lstm_means, width, label='LSTM', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, rnn_means, width, label='RNN', alpha=0.7, color='green')
    
    ax.set_ylabel('Độ chính xác trung bình')
    ax.set_title('Hiệu suất theo loại Optimizer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(unique_optimizers)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/optimizer_effect.png', dpi=300)
    plt.close()

def test_user_input(model_path, config, vocab_dict, pad_idx, max_len=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo mô hình tương ứng
    if config["model_type"] == "RNN":
        model = RNN(
            config["vocab_size"], config["embedding_dim"], config["hidden_size"], 
            config["num_layers"], config["output_size"], pad_idx
        )
    else:  # LSTM
        model = LSTM(
            config["vocab_size"], config["embedding_dim"], config["hidden_size"], 
            config["num_layers"], config["output_size"], pad_idx
        )
    
    # Tải trọng số đã lưu
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Hàm tiền xử lý văn bản
    def preprocess_text(text):
        # Chuyển về chữ thường
        text = text.lower()
        # Loại bỏ ký tự đặc biệt và giữ lại khoảng trắng
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Chuyển thành indices
        indices = []
        for token in tokens:
            if token in vocab_dict:
                indices.append(vocab_dict[token])
            else:
                indices.append(vocab_dict["<unk>"])
        
        # Padding/truncating
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [pad_idx] * (max_len - len(indices))
        
        return torch.tensor(indices).unsqueeze(0)
    
    print("\n===== KIỂM TRA ĐÁNH GIÁ PHIM =====")
    print("(Nhập 'q' để thoát)")
    
    while True:
        user_input = input("\nNhập đánh giá phim của bạn: ")
        if user_input.lower() == 'q':
            break
        
        # Tiền xử lý văn bản nhập vào
        input_tensor = preprocess_text(user_input).to(device)
        
        # Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Hiển thị kết quả
        sentiment = "TÍCH CỰC" if prediction == 1 else "TIÊU CỰC"
        print(f"Phân loại: {sentiment} (độ tin cậy: {confidence:.2%})")

def load_vocab_dict(data_path="dataset/processed_data.pkl"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["vocab"]["<pad>"]

def main():
    # Đảm bảo thư mục kết quả tồn tại
    os.makedirs("results", exist_ok=True)
    
    # Tải kết quả
    results = load_results()
    
    # Vẽ đồ thị so sánh độ chính xác
    plot_accuracy_comparison(results)
    
    # So sánh hiệu suất giữa RNN và LSTM
    plot_model_type_comparison(results)
    
    # Phân tích ảnh hưởng của siêu tham số
    plot_hyperparameter_effects(results)
    
    # Tìm cấu hình tốt nhất cho RNN và LSTM
    best_lstm = max([r for r in results if r["config"]["model_type"] == "LSTM"], 
                    key=lambda x: x["mean_acc"])
    best_rnn = max([r for r in results if r["config"]["model_type"] == "RNN"], 
                   key=lambda x: x["mean_acc"])
    
    print("Cấu hình LSTM tốt nhất:")
    print(json.dumps(best_lstm["config"], indent=2))
    print(f"Độ chính xác trung bình: {best_lstm['mean_acc']:.4f}")
    
    print("\nCấu hình RNN tốt nhất:")
    print(json.dumps(best_rnn["config"], indent=2))
    print(f"Độ chính xác trung bình: {best_rnn['mean_acc']:.4f}")
    
    # Đánh giá chi tiết mô hình tốt nhất
    print("\nĐánh giá mô hình LSTM tốt nhất:")
    lstm_config = best_lstm["config"]
    lstm_model_path = f"checkpoint/best_LSTM_{lstm_config['num_layers']}_{lstm_config['hidden_size']}_{lstm_config['optimizer_type']}.pth"
    
    if os.path.exists(lstm_model_path):
        lstm_acc, lstm_loss, lstm_report = load_model_and_evaluate(lstm_model_path, lstm_config)
        print(f"Độ chính xác: {lstm_acc:.4f}, Loss: {lstm_loss:.4f}")
        print("Classification Report:")
        print(json.dumps(lstm_report, indent=2))
    else:
        print(f"Không tìm thấy file checkpoint: {lstm_model_path}")
    
    print("\nĐánh giá mô hình RNN tốt nhất:")
    rnn_config = best_rnn["config"]
    rnn_model_path = f"checkpoint/best_RNN_{rnn_config['num_layers']}_{rnn_config['hidden_size']}_{rnn_config['optimizer_type']}.pth"
    
    if os.path.exists(rnn_model_path):
        rnn_acc, rnn_loss, rnn_report = load_model_and_evaluate(rnn_model_path, rnn_config)
        print(f"Độ chính xác: {rnn_acc:.4f}, Loss: {rnn_loss:.4f}")
        print("Classification Report:")
        print(json.dumps(rnn_report, indent=2))
    else:
        print(f"Không tìm thấy file checkpoint: {rnn_model_path}")
    
    # Thêm chức năng kiểm tra đánh giá người dùng
    print("\nBạn muốn thử đánh giá phim với mô hình? (y/n): ")
    user_choice = input().lower()
    if user_choice == 'y':
        vocab_dict, pad_idx = load_vocab_dict()
        # Sử dụng mô hình LSTM mặc định vì thường hiệu quả hơn
        if os.path.exists(lstm_model_path):
            test_user_input(lstm_model_path, lstm_config, vocab_dict, pad_idx)
        elif os.path.exists(rnn_model_path):
            test_user_input(rnn_model_path, rnn_config, vocab_dict, pad_idx)
        else:
            print("Không tìm thấy mô hình đã huấn luyện!")

if __name__ == "__main__":
    main()
