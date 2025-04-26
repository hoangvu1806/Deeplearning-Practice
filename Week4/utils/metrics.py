import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import config

def evaluate_predictions(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """Đánh giá hiệu suất dự đoán, tính toán các metrics"""
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }
    
    return metrics

def print_classification_report(true_labels: List[int], pred_labels: List[int]) -> None:
    """In báo cáo phân loại chi tiết"""
    class_names = [config.INV_LABEL_MAP[i] for i in range(len(config.LABEL_MAP))]
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)

def plot_confusion_matrix(true_labels: List[int], pred_labels: List[int], save_path: str = None) -> None:
    """Vẽ confusion matrix"""
    class_names = [config.INV_LABEL_MAP[i] for i in range(len(config.LABEL_MAP))]
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    This class is kept as it represents a significant, potentially complex mechanism.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, path: str = 'best_model.pt'):
        """
        Args:
            patience (int): Số lượng epoch chờ đợi không cải thiện
            verbose (bool): In thông báo
            delta (float): Ngưỡng tối thiểu để coi là cải thiện
            path (str): Đường dẫn để lưu checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Lưu model khi validation loss giảm"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 