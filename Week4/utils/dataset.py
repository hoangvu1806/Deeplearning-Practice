import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from typing import List, Dict, Tuple
import random
from utils.preprocessing import preprocess_text, Vocabulary
import config

class NewsDataset(Dataset):
    """Dataset cho dữ liệu gốc, thực hiện tiền xử lý khi lấy item."""
    def __init__(self, data_items: List[Dict], vocab: Vocabulary, max_seq_len: int):
        self.data_items = data_items
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data_items[idx]
        file_path = item['file_path']
        category = item['category']
        label = config.LABEL_MAP[category]
        
        # Đọc và tiền xử lý văn bản
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        preprocessed_text = preprocess_text(text)
        # Chuyển thành tensor
        tensor = self.vocab.text_to_indices(preprocessed_text, self.max_seq_len)
        
        return tensor, label

class ProcessedNewsDataset(Dataset):
    """Dataset cho dữ liệu đã được tiền xử lý, chỉ cần chuyển đổi sang tensor."""
    def __init__(self, processed_data: List[Dict], vocab: Vocabulary, max_seq_len: int):
        self.processed_data = processed_data
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.processed_data[idx]
        preprocessed_text = item['preprocessed_text']
        category = item['category']
        label = config.LABEL_MAP[category]
        
        # Văn bản đã được tiền xử lý, chỉ cần chuyển thành tensor
        tensor = self.vocab.text_to_indices(preprocessed_text, self.max_seq_len)
        
        return tensor, label

def load_data(mapping_file: str) -> List[Dict]:
    """Tải dữ liệu gốc từ file mapping (JSON)."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_processed_data(data_path: str) -> List[Dict]:
    """Tải dữ liệu đã xử lý từ file pickle."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_processed_vocab(vocab_path: str) -> Vocabulary:
    """Tải từ điển đã xử lý từ file pickle."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def split_data(data: List[Dict], val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia dữ liệu gốc thành tập train, validation và test."""
    random.seed(seed)
    random.shuffle(data)
    
    total_size = len(data)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def create_data_loaders(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], 
                       vocab: Vocabulary, max_seq_len: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo các DataLoader cho dữ liệu gốc (sử dụng NewsDataset)."""
    # Tạo các dataset
    train_dataset = NewsDataset(train_data, vocab, max_seq_len)
    val_dataset = NewsDataset(val_data, vocab, max_seq_len)
    test_dataset = NewsDataset(test_data, vocab, max_seq_len)
    
    # Tạo các dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def create_processed_data_loaders(processed_dir: str, vocab: Vocabulary, 
                                max_seq_len: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo các DataLoader từ dữ liệu đã xử lý (sử dụng ProcessedNewsDataset)."""
    # Tải dữ liệu đã xử lý
    train_data = load_processed_data(os.path.join(processed_dir, 'train_data.pkl'))
    val_data = load_processed_data(os.path.join(processed_dir, 'val_data.pkl'))
    test_data = load_processed_data(os.path.join(processed_dir, 'test_data.pkl'))
    
    # Tạo các dataset
    train_dataset = ProcessedNewsDataset(train_data, vocab, max_seq_len)
    val_dataset = ProcessedNewsDataset(val_data, vocab, max_seq_len)
    test_dataset = ProcessedNewsDataset(test_data, vocab, max_seq_len)
    
    # Tạo các dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def build_vocabulary(data: List[Dict], min_freq: int = 2) -> Vocabulary:
    """Xây dựng từ điển từ dữ liệu gốc."""
    print("Đang xây dựng từ điển...")
    vocab = Vocabulary(min_freq=min_freq)
    texts = []
    
    # Đọc và tiền xử lý tất cả văn bản
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"Đã xử lý {i}/{len(data)} văn bản")
            
        with open(item['file_path'], 'r', encoding='utf-8') as f:
            text = f.read()
        preprocessed_text = preprocess_text(text)
        texts.append(preprocessed_text)
    
    # Xây dựng từ điển
    vocab.build_vocab(texts)
    print(f"Kích thước từ điển: {vocab.vocab_size} từ")
    
    return vocab 