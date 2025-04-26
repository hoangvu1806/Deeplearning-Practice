import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

from utils.dataset import load_data, split_data, create_data_loaders, build_vocabulary
from utils.dataset import load_processed_vocab, create_processed_data_loaders, Vocabulary
from utils.metrics import evaluate_predictions, print_classification_report, plot_confusion_matrix
from models.lstm_attention import LSTMAttention
import config

def evaluate_model(model_path: str = None, vocab_path: str = None, use_processed_data=True):
    """Đánh giá mô hình trên tập test, tính metrics, vẽ confusion matrix và phân tích attention.
    
    Args:
        model_path (str, optional): Đường dẫn đến file trọng số mô hình (.pt). Nếu None, dùng đường dẫn mặc định.
        vocab_path (str, optional): Đường dẫn đến file từ điển (.txt hoặc .pkl).
                                   Chỉ cần thiết nếu use_processed_data=False.
        use_processed_data (bool): True để dùng dữ liệu đã xử lý, False để xử lý trực tiếp.
    
    Returns:
        Dict[str, float]: Dictionary chứa accuracy và f1_macro.
    """
    processed_dir = "processed_data"
    processed_vocab_path = os.path.join(processed_dir, 'vocab.pkl')
    
    if use_processed_data and os.path.exists(processed_vocab_path):
        print("Sử dụng dữ liệu đã xử lý...")
        
        vocab = load_processed_vocab(processed_vocab_path)
        
        
        _, _, test_loader = create_processed_data_loaders(
            processed_dir, vocab, config.MAX_SEQ_LEN, config.BATCH_SIZE
        )
        
        
        info_path = os.path.join(processed_dir, 'dataset_info.json')
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
        
        print(f"Số lượng mẫu test: {dataset_info['test_size']}")
        print(f"Kích thước từ điển: {vocab.vocab_size} từ")
    else:
        print("Sử dụng dữ liệu gốc...")
        
        print("Đang tải dữ liệu...")
        data = load_data(config.MAPPING_FILE)
        
        
        train_data, val_data, test_data = split_data(
            data, 
            val_ratio=config.VALIDATION_SPLIT, 
            test_ratio=config.TEST_SPLIT
        )
        
        
        if vocab_path is None:
            vocab_path = os.path.join(config.MODEL_SAVE_DIR, 'vocab.txt')
        
        print(f"Đang tải từ điển từ {vocab_path}...")
        vocab = Vocabulary(min_freq=2) # Giả sử vocab.txt là loại vocab cũ 
        vocab.load_vocab(vocab_path)  # Cần kiểm tra xem load_vocab có hỗ trợ .txt không
        
        
        _, _, test_loader = create_data_loaders(
            train_data, val_data, test_data,
            vocab, config.MAX_SEQ_LEN, config.BATCH_SIZE
        )
    
    
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
    
    
    true_labels = []
    pred_labels = []
    attention_weights_list = []
    test_texts = []
    
    print("Đang đánh giá mô hình...")
    with torch.no_grad():
        for batch in test_loader:
            text, labels = batch
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)
            
            predictions, attention_weights = model(text)
            
            _, preds = torch.max(predictions, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            
            
            attention_weights_list.extend(attention_weights.cpu().numpy())
            test_texts.extend(text.cpu().numpy()) 
    
    
    metrics = evaluate_predictions(true_labels, pred_labels)
    print(f"\nKết quả đánh giá:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Macro: {metrics['f1_macro']:.4f}")
    
    
    print("\nBáo cáo phân loại chi tiết:")
    print_classification_report(true_labels, pred_labels)
    
    
    cm_path = os.path.join(config.MODEL_SAVE_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels, pred_labels, save_path=cm_path)
    
    
    results = {
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
    }
    
    with open(os.path.join(config.MODEL_SAVE_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f)
    
    print(f"\nĐã lưu kết quả đánh giá và confusion matrix tại {config.MODEL_SAVE_DIR}")
    
    
    analyze_attention_weights(test_texts, attention_weights_list, true_labels, pred_labels, vocab)
    
    return metrics

def analyze_attention_weights(texts, attention_weights, true_labels, pred_labels, vocab, num_samples=5):
    """Phân tích và trực quan hóa attention weights trên một số mẫu ngẫu nhiên từ tập test."""
    
    np.random.seed(42)
    total_samples = len(texts)
    indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    
    
    analysis_dir = os.path.join(config.MODEL_SAVE_DIR, 'attention_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\nPhân tích Attention Weights cho {num_samples} mẫu...")
    for i, idx in enumerate(indices):
        text = texts[idx]
        attn_weights = attention_weights[idx]
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        
        
        tokens = []
        for token_idx in text:
            if token_idx in vocab.idx2token and token_idx != vocab.pad_idx:
                tokens.append(vocab.idx2token[token_idx])
        
        
        attn_values = attn_weights[:len(tokens)] 
        
        
        if len(tokens) > 0 and len(attn_values) == len(tokens):
            sorted_indices = np.argsort(attn_values.flatten())[::-1]
            top_tokens = [(tokens[j], float(attn_values[j])) for j in sorted_indices[:10]]
        else:
            top_tokens = []
        
        
        analysis = {
            'sample_index_in_test_set': int(idx),
            'true_label': config.INV_LABEL_MAP[true_label],
            'predicted_label': config.INV_LABEL_MAP[pred_label],
            'correct': true_label == pred_label,
            'top_10_attended_tokens': top_tokens,
            'original_tokens': tokens # Thêm tokens gốc để dễ xem
        }
        
        with open(os.path.join(analysis_dir, f'sample_{i+1}.json'), 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        
        if len(tokens) > 0:
            plt.figure(figsize=(12, 6))
            
            
            max_display_tokens = 50
            display_tokens = tokens[:max_display_tokens]
            display_weights = attn_values.flatten()[:max_display_tokens]
            
            plt.bar(range(len(display_tokens)), display_weights)
            plt.xticks(range(len(display_tokens)), display_tokens, rotation=90)
            plt.title(f"Sample {i+1}: True={config.INV_LABEL_MAP[true_label]}, Pred={config.INV_LABEL_MAP[pred_label]}")
            plt.ylabel("Attention Weight")
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'sample_{i+1}_attention.png'))
            plt.close()
        else:
            print(f"Bỏ qua vẽ biểu đồ cho sample {i+1} vì không có token nào sau khi loại bỏ padding.")
    
    print(f"Đã lưu phân tích attention weights cho {num_samples} mẫu tại {analysis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Đánh giá mô hình phân loại tin tức tiếng Việt')
    parser.add_argument('--raw', action='store_true', help='Sử dụng dữ liệu thô thay vì dữ liệu đã xử lý')
    parser.add_argument('--model', type=str, help='Đường dẫn đến file mô hình (.pt)')
    parser.add_argument('--vocab', type=str, help='Đường dẫn đến file từ điển (cần nếu dùng --raw)')
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model, vocab_path=args.vocab, use_processed_data=not args.raw) 