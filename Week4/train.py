import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import argparse

from utils.dataset import load_data, split_data, create_data_loaders, build_vocabulary
from utils.dataset import load_processed_vocab, create_processed_data_loaders
from utils.preprocessing import Vocabulary
from utils.metrics import EarlyStopping, evaluate_predictions
from models.lstm_attention import LSTMAttention
import config

def train_model(use_processed_data=True):
    processed_dir = "processed_data"
    vocab_path = os.path.join(processed_dir, 'vocab.pkl')
    
    if use_processed_data and os.path.exists(vocab_path):
        print("Sử dụng dữ liệu đã xử lý...")
        
        vocab = load_processed_vocab(vocab_path)
        
        
        train_loader, val_loader, test_loader = create_processed_data_loaders(
            processed_dir, vocab, config.MAX_SEQ_LEN, config.BATCH_SIZE
        )
        
        
        info_path = os.path.join(processed_dir, 'dataset_info.json')
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
        
        print(f"Số lượng mẫu: Train: {dataset_info['train_size']}, "
              f"Val: {dataset_info['val_size']}, Test: {dataset_info['test_size']}")
        print(f"Kích thước từ điển: {vocab.vocab_size} từ")
        
    else:
        print("Không tìm thấy dữ liệu đã xử lý hoặc đã chọn xử lý trực tiếp...")
        
        print("Đang tải dữ liệu...")
        data = load_data(config.MAPPING_FILE)
        
        
        train_data, val_data, test_data = split_data(
            data, 
            val_ratio=config.VALIDATION_SPLIT, 
            test_ratio=config.TEST_SPLIT
        )
        
        print(f"Số lượng mẫu: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        
        vocab_path_raw = os.path.join(config.MODEL_SAVE_DIR, 'vocab.txt')
        if os.path.exists(vocab_path_raw):
            print("Đang tải từ điển có sẵn...")
            vocab = Vocabulary(min_freq=2)
            vocab.load_vocab(vocab_path_raw)
        else:
            print("Đang xây dựng từ điển từ tập train...")
            vocab = build_vocabulary(train_data, min_freq=2)
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            vocab.save_vocab(vocab_path_raw)
        
        
        train_loader, val_loader, test_loader = create_data_loaders(
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
    
    
    model = model.to(config.DEVICE)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=model_save_path
    )
    
    
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    
    print("Bắt đầu huấn luyện...")
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        
        model.train()
        epoch_loss = 0
        true_labels = []
        pred_labels = []
        
        for batch in train_loader:
            
            text, labels = batch
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions, _ = model(text)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * text.size(0)
            _, preds = torch.max(predictions, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        train_metric = evaluate_predictions(true_labels, pred_labels)
        train_metrics.append(train_metric)
        
        
        model.eval()
        val_loss = 0
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                text, labels = batch
                text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)
                
                predictions, _ = model(text)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item() * text.size(0)
                
                _, preds = torch.max(predictions, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_metric = evaluate_predictions(true_labels, pred_labels)
        val_metrics.append(val_metric)
        scheduler.step()

        print(f'Epoch: {epoch}/{config.EPOCHS}')
        print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {train_metric["accuracy"]:.4f}, F1: {train_metric["f1_macro"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_metric["accuracy"]:.4f}, F1: {val_metric["f1_macro"]:.4f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    training_time = time.time() - start_time
    print(f'Hoàn thành huấn luyện trong {training_time/60:.2f} phút')
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': [m['accuracy'] for m in train_metrics],
        'val_accuracy': [m['accuracy'] for m in val_metrics],
        'train_f1': [m['f1_macro'] for m in train_metrics],
        'val_f1': [m['f1_macro'] for m in val_metrics],
    }
    
    with open(os.path.join(config.MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot([m['accuracy'] for m in train_metrics], label='Train Accuracy')
    plt.plot([m['accuracy'] for m in val_metrics], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_DIR, 'training_curves.png'))
    plt.show()
    
    return model, vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại tin tức tiếng Việt')
    parser.add_argument('--raw', action='store_true', help='Sử dụng dữ liệu thô thay vì dữ liệu đã xử lý')
    args = parser.parse_args()
    train_model(use_processed_data=not args.raw)