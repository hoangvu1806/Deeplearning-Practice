import torch
import os

DATA_DIR = "data/news"
MAPPING_FILE = "data/mapping_data.json"
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Cấu hình dữ liệu
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Cấu hình mô hình
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
ATTENTION_DIM = 64

# Cấu hình huấn luyện
LEARNING_RATE = 0.001
EPOCHS = 10 
EARLY_STOPPING_PATIENCE = 5

# Cấu hình thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Danh sách thể loại
CATEGORIES = [os.path.basename(os.path.normpath(f.path)) 
              for f in os.scandir(DATA_DIR) if f.is_dir()]

# Ánh xạ nhãn
LABEL_MAP = {cat: idx for idx, cat in enumerate(sorted(CATEGORIES))}
INV_LABEL_MAP = {idx: cat for cat, idx in LABEL_MAP.items()} 