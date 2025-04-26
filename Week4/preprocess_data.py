import json
import os
import pickle
from tqdm import tqdm
import numpy as np

from utils.preprocessing import preprocess_text, Vocabulary
from utils.dataset import load_data, split_data
import config

def preprocess_and_save_data():
    """Tải dữ liệu gốc, tiền xử lý (làm sạch, tách từ), chia tập,
    xây dựng từ điển và lưu tất cả vào thư mục processed_data.
    """
    
    processed_dir = "processed_data"
    os.makedirs(processed_dir, exist_ok=True)
    
    
    print("Đang tải dữ liệu...")
    data = load_data(config.MAPPING_FILE)
    
    
    train_data, val_data, test_data = split_data(
        data, 
        val_ratio=config.VALIDATION_SPLIT, 
        test_ratio=config.TEST_SPLIT,
        seed=42
    )
    
    print(f"Số lượng mẫu: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    processed_datasets = {}
    all_texts = []
    
    for name, dataset in datasets.items():
        print(f"Đang xử lý tập {name}...")
        processed_data = []
        
        for item in tqdm(dataset):
            file_path = item['file_path']
            category = item['category']
            
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            preprocessed_text = preprocess_text(text)
            
            
            processed_item = {
                'preprocessed_text': preprocessed_text,
                'category': category,
                'original_path': file_path
            }
            
            processed_data.append(processed_item)
            all_texts.append(preprocessed_text)
        
        processed_datasets[name] = processed_data
    
    
    print("Đang xây dựng từ điển từ tất cả các tập...")
    vocab = Vocabulary(min_freq=2) 
    vocab.build_vocab(all_texts)
    print(f"Kích thước từ điển: {vocab.vocab_size} từ")
    
    
    vocab_path = os.path.join(processed_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    
    for name, processed_data in processed_datasets.items():
        output_path = os.path.join(processed_dir, f'{name}_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
    
    
    dataset_info = {
        'train_size': len(processed_datasets['train']),
        'val_size': len(processed_datasets['val']),
        'test_size': len(processed_datasets['test']),
        'vocab_size': vocab.vocab_size
    }
    
    info_path = os.path.join(processed_dir, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)
    
    print("Đã hoàn thành tiền xử lý và lưu dữ liệu:")
    print(f"- Từ điển: {vocab_path}")
    for name in processed_datasets:
        print(f"- Tập {name}: {os.path.join(processed_dir, f'{name}_data.pkl')}")
    print(f"- Thông tin dataset: {info_path}")

if __name__ == "__main__":
    preprocess_and_save_data() 