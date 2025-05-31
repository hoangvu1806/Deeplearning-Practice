import os
import re
import pickle
import nltk
import numpy as np
from collections import Counter
from tqdm import tqdm
from underthesea import word_tokenize as vi_tokenize
from nltk.tokenize import word_tokenize as en_tokenize

nltk.download('punkt', quiet=True)
import config

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vocab(sentences, max_size, min_freq):
    counter = Counter()
    for sentence in tqdm(sentences, desc="Đếm từ"):
        counter.update(sentence)
    
    # Lọc từ có tần suất >= min_freq
    filtered = [(word, count) for word, count in counter.items() if count >= min_freq]
    most_common = sorted(filtered, key=lambda x: x[1], reverse=True)[:max_size - len(config.SPECIAL_TOKENS)]
    vocab = {word: idx + len(config.SPECIAL_TOKENS) for idx, (word, _) in enumerate(most_common)}
    
    for idx, token in enumerate(config.SPECIAL_TOKENS):
        vocab[token] = idx
    
    return vocab

def encode_sentence(sentence, vocab, max_length):
    tokens = [config.SOS_TOKEN] + sentence + [config.EOS_TOKEN]
    indices = [vocab.get(token, vocab[config.UNK_TOKEN]) for token in tokens]
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices += [vocab[config.PAD_TOKEN]] * (max_length - len(indices))
    return indices

def load_and_process_data():
    print("Đang tải dữ liệu...")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    with open(config.EN_DATA_PATH, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    with open(config.VI_DATA_PATH, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]
    
    # Lọc bỏ các dòng có chứa từ "tom"
    filtered_pairs = [(en, vi) for en, vi in zip(en_lines, vi_lines) if 'tom' not in en.lower()]
    en_lines = [pair[0] for pair in filtered_pairs]
    vi_lines = [pair[1] for pair in filtered_pairs]
    
    assert len(en_lines) == len(vi_lines), "Số lượng câu tiếng Anh và tiếng Việt không khớp!"
    print(f"Đã tải {len(en_lines)} cặp câu.")
    
    print("Đang làm sạch và chuẩn hóa dữ liệu...")
    en_cleaned = [clean_text(line) for line in tqdm(en_lines, desc="Làm sạch EN")]
    vi_cleaned = [clean_text(line) for line in tqdm(vi_lines, desc="Làm sạch VI")]
    
    print("Đang tokenize...")
    en_tokenized = [en_tokenize(sentence) for sentence in tqdm(en_cleaned, desc="Tokenize EN")]
    vi_tokenized = [vi_tokenize(sentence) for sentence in tqdm(vi_cleaned, desc="Tokenize VI")]
    
    print("Đang xây dựng từ điển...")
    en_vocab = build_vocab(en_tokenized, config.MAX_VOCAB_SIZE, config.MIN_FREQ)
    vi_vocab = build_vocab(vi_tokenized, config.MAX_VOCAB_SIZE, config.MIN_FREQ)
    
    print(f"Kích thước từ điển tiếng Anh: {len(en_vocab)}")
    print(f"Kích thước từ điển tiếng Việt: {len(vi_vocab)}")
    
    with open(config.VOCAB_EN_PATH, 'wb') as f:
        pickle.dump(en_vocab, f)
    with open(config.VOCAB_VI_PATH, 'wb') as f:
        pickle.dump(vi_vocab, f)
    
    print("Đang mã hóa câu...")
    en_encoded = [encode_sentence(sentence, en_vocab, config.MAX_SEQUENCE_LENGTH) 
                 for sentence in tqdm(en_tokenized, desc="Mã hóa EN")]
    vi_encoded = [encode_sentence(sentence, vi_vocab, config.MAX_SEQUENCE_LENGTH) 
                 for sentence in tqdm(vi_tokenized, desc="Mã hóa VI")]
    
    # Kiểm tra tỷ lệ <unk>
    for lang, encoded, vocab in [("tiếng Anh", en_encoded, en_vocab), ("tiếng Việt", vi_encoded, vi_vocab)]:
        unk_count = sum(indices.count(vocab[config.UNK_TOKEN]) for indices in encoded)
        total_tokens = sum(len(indices) for indices in encoded)
        print(f"Tỷ lệ <unk> {lang}: {unk_count / total_tokens:.2%}")
    
    en_data = np.array(en_encoded)
    vi_data = np.array(vi_encoded)
    
    print("Đang chia dữ liệu thành train/val/test...")
    indices = np.random.permutation(len(en_data))
    
    train_size = int(len(indices) * config.TRAIN_RATIO)
    val_size = int(len(indices) * config.VAL_RATIO)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = (en_data[train_indices], vi_data[train_indices])
    val_data = (en_data[val_indices], vi_data[val_indices])
    test_data = (en_data[test_indices], vi_data[test_indices])
    
    with open(config.TRAIN_DATA_PATH, 'wb') as f:
        pickle.dump(train_data, f)
    with open(config.VAL_DATA_PATH, 'wb') as f:
        pickle.dump(val_data, f)
    with open(config.TEST_DATA_PATH, 'wb') as f:
        pickle.dump(test_data, f)
    
    print("Tiền xử lý dữ liệu hoàn tất!")
    print(f"Số lượng mẫu train: {len(train_indices)}")
    print(f"Số lượng mẫu validation: {len(val_indices)}")
    print(f"Số lượng mẫu test: {len(test_indices)}")
    
    return en_vocab, vi_vocab, train_data, val_data, test_data

if __name__ == "__main__":
    load_and_process_data()