import torch
import argparse
import os
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_text, Vocabulary
from utils.dataset import load_processed_vocab
from models.lstm_attention import LSTMAttention
import config

def load_model_and_vocab(model_path: str = None, vocab_path: str = None, use_processed_data=True) -> Tuple[LSTMAttention, Vocabulary]:
    """Tải mô hình LSTM+Attention đã huấn luyện và từ điển tương ứng."""
    
    processed_dir = "processed_data"
    processed_vocab_path = os.path.join(processed_dir, 'vocab.pkl')
    
    if use_processed_data and os.path.exists(processed_vocab_path):
        print("Sử dụng từ điển đã xử lý...")
        vocab = load_processed_vocab(processed_vocab_path)
    else:
        print("Sử dụng từ điển gốc...")
        if vocab_path is None:
            vocab_path = os.path.join(config.MODEL_SAVE_DIR, 'vocab.txt')
        print(f"Đang tải từ điển từ {vocab_path}...")
        vocab = Vocabulary(min_freq=2) # Giả sử min_freq=2 khi tạo vocab
        vocab.load_vocab(vocab_path) 
    
    
    model = LSTMAttention(
        vocab_size=vocab.vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        output_dim=len(config.LABEL_MAP),
        num_layers=config.NUM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        dropout=config.DROPOUT,
        pad_idx=vocab.pad_idx
    )
    
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
    
    print(f"Đang tải mô hình từ {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    return model, vocab

def predict_text(text: str, model: LSTMAttention, vocab: Vocabulary) -> Tuple[str, Dict, np.ndarray]:
    """Dự đoán thể loại cho một đoạn văn bản và trả về attention weights."""
    
    processed_text = preprocess_text(text)
    
    
    text_tensor = vocab.text_to_indices(processed_text, config.MAX_SEQ_LEN).unsqueeze(0).to(config.DEVICE)
    
    
    with torch.no_grad():
        predictions, attention_weights = model(text_tensor)
        
        
        probs = torch.softmax(predictions, dim=1)
        pred_class = torch.argmax(predictions, dim=1).item()
        
        
        all_probs = {config.INV_LABEL_MAP[i]: float(probs[0, i]) for i in range(len(config.LABEL_MAP))}
        
        
        predicted_category = config.INV_LABEL_MAP[pred_class]
        
        
        attention_weights = attention_weights.cpu().numpy()
    
    return predicted_category, all_probs, attention_weights

def visualize_attention(text: str, attention_weights: np.ndarray, vocab: Vocabulary, save_path: str = None):
    """Trực quan hóa attention weights dưới dạng biểu đồ cột."""
    
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    
    if not tokens:
        print("Không thể trực quan hóa attention: không có token nào sau tiền xử lý.")
        return
        
    
    attention_values = attention_weights[0, :len(tokens)]
    
    if len(attention_values) != len(tokens):
         print(f"Cảnh báo: Độ dài attention weights ({len(attention_values)}) không khớp với số lượng tokens ({len(tokens)}). Biểu đồ có thể không chính xác.")
         # Cố gắng khớp độ dài nếu có thể
         min_len = min(len(attention_values), len(tokens))
         tokens = tokens[:min_len]
         attention_values = attention_values[:min_len]
         if not tokens:
             print("Không thể trực quan hóa attention sau khi khớp độ dài.")
             return

    
    plt.figure(figsize=(12, 6))
    
    
    max_display_tokens = 50
    display_tokens = tokens[:max_display_tokens]
    display_weights = attention_values.flatten()[:max_display_tokens]
    
    plt.bar(range(len(display_tokens)), display_weights)
    plt.xticks(range(len(display_tokens)), display_tokens, rotation=90)
    plt.ylabel("Attention Weight")
    plt.title("Attention Weights on Input Text")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ attention tại: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Dự đoán thể loại cho văn bản tiếng Việt')
    parser.add_argument('--text', type=str, help='Văn bản đầu vào trực tiếp')
    parser.add_argument('--file', type=str, help='Đường dẫn đến file .txt chứa văn bản')
    parser.add_argument('--visualize', action='store_true', help='Trực quan hóa attention weights')
    parser.add_argument('--raw', action='store_true', help='Sử dụng từ điển thô (vocab.txt) thay vì từ điển đã xử lý (vocab.pkl)')
    parser.add_argument('--model', type=str, help='Đường dẫn đến file mô hình (.pt)')
    parser.add_argument('--vocab', type=str, help='Đường dẫn đến file từ điển (cần nếu dùng --raw)')
    parser.add_argument('--save_plot', type=str, help='Đường dẫn để lưu biểu đồ attention (nếu dùng --visualize)')
    args = parser.parse_args()
    
    
    model, vocab = load_model_and_vocab(model_path=args.model, vocab_path=args.vocab, use_processed_data=not args.raw)
    
    
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại {args.file}")
            return
        except Exception as e:
             print(f"Lỗi khi đọc file {args.file}: {e}")
             return
    else:
        
        print("Không có văn bản đầu vào từ --text hoặc --file.")
        print("Nhập văn bản của bạn (nhấn Enter 2 lần để kết thúc):")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
        except EOFError:
            print("\nĐã hủy nhập.")
            return
    
    if not text.strip():
        print("Văn bản đầu vào trống. Kết thúc.")
        return
    
    
    predicted_category, all_probs, attention_weights = predict_text(text, model, vocab)
    
    
    print(f"\nThể loại dự đoán: {predicted_category}")
    print("\nXác suất các thể loại:")
    for category, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"- {category}: {prob:.4f}")
    
    
    if args.visualize:
        visualize_attention(text, attention_weights, vocab, save_path=args.save_plot)

if __name__ == "__main__":
    main() 