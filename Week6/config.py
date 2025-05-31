SEED = 18
# Đường dẫn dữ liệu
EN_DATA_PATH = "data/en_sents.txt"
VI_DATA_PATH = "data/vi_sents.txt"

# Cấu hình tiền xử lý
MAX_VOCAB_SIZE = 30000  # Tăng từ 10000
MIN_FREQ = 2  # Từ phải xuất hiện ít nhất 2 lần để vào từ điển
MAX_SEQUENCE_LENGTH = 15
TRAIN_RATIO = 0.8  
VAL_RATIO = 0.1  

# Cấu hình đường dẫn lưu dữ liệu đã xử lý
PROCESSED_DIR = "data/processed"
VOCAB_EN_PATH = f"{PROCESSED_DIR}/vocab_en.pkl"  
VOCAB_VI_PATH = f"{PROCESSED_DIR}/vocab_vi.pkl"
TRAIN_DATA_PATH = f"{PROCESSED_DIR}/train_data.pkl"
VAL_DATA_PATH = f"{PROCESSED_DIR}/val_data.pkl"
TEST_DATA_PATH = f"{PROCESSED_DIR}/test_data.pkl"

# Token đặc biệt
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"  # Start of sentence
EOS_TOKEN = "<eos>"  # End of sentence

# Danh sách token đặc biệt
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] 


# Các cấu hình siêu tham số cho mô hình RNN (Seq2Seq với Attention)
RNN_CONFIGS = [
    {
        'name': 'RNN_Config_1',
        'embedding_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.003,
        'batch_size': 128,
        'num_epochs': 15,
        'clip': 1.0
    },
    {
        'name': 'RNN_Config_2',
        'embedding_size': 128,
        'hidden_size': 256,
        'num_layers': 1,
        'dropout': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.005,
        'batch_size': 256,
        'num_epochs': 15,
        'clip': 1.0
    },
    {
        'name': 'RNN_Config_3',
        'embedding_size': 128,
        'hidden_size': 64,
        'num_layers': 4,
        'dropout': 0.4,
        'optimizer': 'sgd',
        'learning_rate': 0.003,
        'batch_size': 128,
        'num_epochs': 15,
        'clip': 0.5
    },
    {
        'name': 'RNN_Config_4',
        'embedding_size': 256,
        'hidden_size': 64,
        'num_layers': 3,
        'dropout': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 256,
        'num_epochs': 15,
        'clip': 1.0
    },
    {
        'name': 'RNN_Config_5',
        'embedding_size': 128,
        'hidden_size': 64,
        'num_layers': 4,
        'dropout': 0.5,
        'optimizer': 'adam',
        'learning_rate': 0.0025,
        'batch_size': 128,
        'num_epochs': 15,
        'clip': 1.0
    }
]

# Các cấu hình siêu tham số cho mô hình Transformer (Fine-tuning)
TRANSFORMER_CONFIGS = [
    {
        'name': 'Transformer_Fine-tune_1',
        'model_name': 'Helsinki-NLP/opus-mt-en-vi',
        'optimizer': 'adamw',
        'num_epochs': 5,
        'learning_rate': 5e-5,
        'batch_size': 128,
    },
    {
        'name': 'Transformer_Fine-tune_2',
        'model_name': 'Helsinki-NLP/opus-mt-en-vi',
        'optimizer': 'adamw',
        'num_epochs': 5,
        'learning_rate': 3e-5,
        'batch_size': 64,
    },
    {
        'name': 'Transformer_Fine-tune_3',
        'model_name': 'Helsinki-NLP/opus-mt-en-vi',
        'optimizer': 'adamw',
        'num_epochs': 5,
        'learning_rate': 2e-5,
        'batch_size': 128,
    },
    {
        'name': 'Transformer_Fine-tune_4',
        'model_name': 'Helsinki-NLP/opus-mt-en-vi',
        'optimizer': 'sgd',
        'num_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 64,
    },
    {
        'name': 'Transformer_Fine-tune_5',
        'model_name': 'Helsinki-NLP/opus-mt-en-vi',
        'optimizer': 'adam',
        'num_epochs': 5,
        'learning_rate': 1e-5,
        'batch_size': 64,
    }
]