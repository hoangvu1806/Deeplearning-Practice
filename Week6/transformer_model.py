import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Cache directory
CACHE_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(CACHE_DIR, exist_ok=True)

class TranslationDataset(Dataset):
    def __init__(self, en_file, vi_file, tokenizer, max_length=20):  # Tăng max_length lên 20
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_lines = [line.strip() for line in f if line.strip()]
        with open(vi_file, 'r', encoding='utf-8') as f:
            self.vi_lines = [line.strip() for line in f if line.strip()]
        assert len(self.en_lines) == len(self.vi_lines), "Number of English and Vietnamese sentences do not match!"
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.en_lines)
    
    def __getitem__(self, idx):
        en_text = self.en_lines[idx]
        vi_text = self.vi_lines[idx]
        
        inputs = self.tokenizer(en_text, max_length=self.max_length, 
                               padding='max_length', truncation=True, 
                               return_tensors='pt')
        labels = self.tokenizer(vi_text, max_length=self.max_length, 
                               padding='max_length', truncation=True, 
                               return_tensors='pt')
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def calculate_bleu(model, data_loader, tokenizer):
    model.eval()
    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method3  # Dùng method3 thay vì method1
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating BLEU"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20,  # Đồng bộ với max_length
                num_beams=4,   # Tăng num_beams để cải thiện chất lượng
                early_stopping=True
            )
            for i in range(input_ids.size(0)):
                ref = tokenizer.decode(batch['labels'][i], skip_special_tokens=True).split()
                hyp = tokenizer.decode(outputs[i], skip_special_tokens=True).split()
                if hyp:
                    references.append([ref])
                    hypotheses.append(hyp)
    
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing) if references else 0.0
    return bleu_score

def train_epoch(model, data_loader, optimizer, scaler, config_dict):
    model.train()
    epoch_loss = 0
    accum_steps = config_dict.get('accum_steps', 2)  # Gradient accumulation
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           labels=labels)
            loss = outputs.loss / accum_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accum_steps
    
    # Final step nếu số batch không chia hết cho accum_steps
    if (i + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast('cuda'):
                outputs = model(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               labels=labels)
                loss = outputs.loss
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def train_model(config_dict, train_loader, val_loader, test_loader, tokenizer):
    model_name = config_dict['name']
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config_dict['model_name'], cache_dir=CACHE_DIR
        ).to(device)
        # Tối ưu mô hình nếu PyTorch >= 2.0
        if torch.__version__ >= '2.0':
            model = torch.compile(model)
    except Exception as e:
        logger.error(f"Failed to load model {config_dict['model_name']}: {str(e)}")
        raise
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config_dict['learning_rate']) if config_dict['optimizer'] == 'adamw' else \
                optim.SGD(model.parameters(), 
                         lr=config_dict['learning_rate'])
    
    scaler = GradScaler('cuda')
    
    checkpoint_dir = f"models/{model_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log = {'train_loss': [], 'val_loss': [], 'bleu': []}
    best_val_loss = float('inf')
    patience = 3  # Tăng patience lên 3
    
    # Kiểm tra <unk> ratio
    unk_token_id = tokenizer.unk_token_id
    unk_count = 0
    total_tokens = 0
    for batch in train_loader:
        labels = batch['labels'].flatten()
        unk_count += (labels == unk_token_id).sum().item()
        total_tokens += labels.numel()
    logger.info(f"<unk> ratio in training data: {unk_count / total_tokens:.2%}")
    
    for epoch in range(config_dict['num_epochs']):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, config_dict)
        val_loss = evaluate(model, val_loader)
        
        # Chỉ tính BLEU ở epoch cuối hoặc mỗi 2 epoch
        if epoch % 2 == 0 or epoch == config_dict['num_epochs'] - 1:
            bleu_score = calculate_bleu(model, val_loader, tokenizer)
        else:
            bleu_score = 0.0
        
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['bleu'].append(bleu_score)
        
        if epoch == config_dict['num_epochs'] - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu': bleu_score
            }
            torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch+1}.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu': bleu_score
            }
            torch.save(checkpoint, f"models/{model_name}/best_model.pt")
        
        logger.info(f"Epoch {epoch+1}/{config_dict['num_epochs']} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"BLEU: {bleu_score:.4f} | Time: {time.time() - start_time:.2f}s")
    
    test_loss = evaluate(model, test_loader)
    test_bleu = calculate_bleu(model, test_loader, tokenizer)
    logger.info(f"Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(log['train_loss'], label='Train Loss')
    plt.plot(log['val_loss'], label='Val Loss')
    plt.plot(log['bleu'], label='BLEU Score')
    plt.title(f"Training Metrics - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"models/{model_name}/metrics_plot.png")
    plt.close()
    
    return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss, 
            'bleu': test_bleu, 'log': log}

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.TRANSFORMER_CONFIGS[0]['model_name'], cache_dir=CACHE_DIR, use_fast=False
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {config.TRANSFORMER_CONFIGS[0]['model_name']}: {str(e)}")
        raise
    
    dataset = TranslationDataset(config.EN_DATA_PATH, config.VI_DATA_PATH, 
                                tokenizer)
    
    indices = np.random.permutation(len(dataset))
    train_size = int(len(indices) * config.TRAIN_RATIO)
    val_size = int(len(indices) * config.VAL_RATIO)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    results = []
    
    for cfg in config.TRANSFORMER_CONFIGS:
        logger.info(f"\nTraining {cfg['name']}...")
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                                 shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], 
                               shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], 
                                shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
        
        result = train_model(cfg, train_loader, val_loader, test_loader, tokenizer)
        results.append({
            'name': cfg['name'],
            'train_loss': result['train_loss'],
            'val_loss': result['val_loss'],
            'test_loss': result['test_loss'],
            'bleu': result['bleu']
        })
    
    metrics = ['train_loss', 'val_loss', 'test_loss', 'bleu']
    analysis = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        analysis[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    with open(f"{results_dir}/transformer_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=4)
    
    plt.figure(figsize=(12, 6))
    means = [analysis[metric]['mean'] for metric in metrics]
    stds = [analysis[metric]['std'] for metric in metrics]
    plt.bar([f"{m}_mean" for m in metrics], means, yerr=stds, capsize=5)
    plt.title("Comparison of Metrics Across Transformer Configurations")
    plt.ylabel("Value")
    plt.savefig(f"{results_dir}/transformer_comparison_plot.png")
    plt.close()
    
    logger.info("\nTraining completed. Results saved in 'results' directory.")

if __name__ == "__main__":
    main()