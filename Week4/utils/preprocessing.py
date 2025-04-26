import unicodedata
import re
import torch
import numpy as np
from underthesea import word_tokenize
from typing import List, Dict, Tuple, Set

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', ' ', text) #  bỏ các ký tự HTML
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text) #  bỏ URL
    
    # giữ chữ cái tiếng việt, số, dấu gạch dưới và khoảng trắng
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text: str) -> str:
    text = normalize_unicode(text)
    text = clean_text(text)
    text = word_tokenize(text, format="text")
    return text

class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_counts: Dict[str, int] = {}
        self.min_freq = min_freq
        
        self.pad_token, self.unk_token = "<PAD>", "<UNK>"
        self.pad_idx, self.unk_idx = 0, 1
        self.token2idx[self.pad_token] = self.pad_idx
        self.token2idx[self.unk_token] = self.unk_idx
        self.idx2token[self.pad_idx] = self.pad_token
        self.idx2token[self.unk_idx] = self.unk_token
        self.vocab_size = 2
        
    def build_vocab(self, texts: List[str]) -> None:
        for text in texts:
            for token in text.split():
                if token in self.token_counts:
                    self.token_counts[token] += 1
                else:
                    self.token_counts[token] = 1
        
        
        for token, count in self.token_counts.items():
            if count >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = self.vocab_size
                self.idx2token[self.vocab_size] = token
                self.vocab_size += 1
    
    def text_to_indices(self, text: str, max_len: int) -> torch.Tensor:
        tokens = text.split()
        indices = []
        
        for token in tokens[:max_len]:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)
        
        
        if len(indices) < max_len:
            indices.extend([self.pad_idx] * (max_len - len(indices)))
            
        return torch.tensor(indices, dtype=torch.long)
    
    def save_vocab(self, path: str) -> None:
        """Lưu từ điển (token sang index) xuống file."""
        with open(path, 'w', encoding='utf-8') as f:
            for token, idx in self.token2idx.items():
                f.write(f"{token}\t{idx}\n")
    
    def load_vocab(self, path: str) -> None:
        """Tải từ điển từ file."""
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_size = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                idx = int(idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                self.vocab_size = max(self.vocab_size, idx + 1)
                
        
        self.pad_idx = self.token2idx.get(self.pad_token, 0)
        self.unk_idx = self.token2idx.get(self.unk_token, 1) 