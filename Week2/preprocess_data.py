import pandas as pd
import torch
import os
from tqdm import tqdm
import logging
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from collections import Counter

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IMDbPreprocessor:
    def __init__(self, data_path, vocab_size=15000, max_len=256, output_dir="dataset"):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.output_dir = output_dir
        self.tokenizer = word_tokenize
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}
        self.vocab = None
        self.vocab_dict = {}  # Lưu ánh xạ token -> index
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        logger.info(f"Đang đọc dữ liệu từ {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            return (
                df["review"].tolist(),
                df["sentiment"].apply(lambda x: 1 if x == "positive" else 0).tolist(),
            )
        except FileNotFoundError:
            logger.error(f"File {self.data_path} không tồn tại")
            raise

    def preprocess_text(self, text):
        tokens = self.tokenizer(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()
        ]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def build_vocabulary(self, texts):
        logger.info("Đang xây dựng từ vựng...")

        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)

        token_counts = Counter(all_tokens)
        most_common_tokens = token_counts.most_common(
            self.vocab_size - 2
        )  # Dũ trừ 2 vì có <unk> và <pad>
        vocab_tokens = [self.unk_token, self.pad_token] + [
            token for token, _ in most_common_tokens
        ]

        # Tạo ánh xạ token -> index
        self.vocab_dict = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.vocab = vocab_tokens

        vocab_path = os.path.join(self.output_dir, "vocab.csv")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(f"{token}\n")
        logger.info(f"Đã lưu từ vựng tại {vocab_path}")

    def text_to_indices(self, text):
        tokens = self.preprocess_text(text)[: self.max_len]
        indices = [
            self.vocab_dict.get(token, self.vocab_dict[self.unk_token])
            for token in tokens
        ]

        indices = (
            indices + [self.vocab_dict[self.pad_token]] * (self.max_len - len(indices))
            if len(indices) < self.max_len
            else indices[: self.max_len]
        )
        return indices

    def process_and_save(self):
        texts, labels = self.load_data()

        if self.vocab is None:
            self.build_vocabulary(texts)

        logger.info("Đang chuyển văn bản thành số...")
        processed_texts = []
        for text in tqdm(texts, desc="Processing texts"):
            indices = self.text_to_indices(text)
            processed_texts.append(indices)

        texts_tensor = torch.tensor(processed_texts, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        output_path = os.path.join(self.output_dir, "processed_data.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "texts": texts_tensor,
                    "labels": labels_tensor,
                    "vocab": self.vocab_dict,
                },
                f,
            )
        logger.info(f"Đã lưu dữ liệu đã xử lý tại {output_path}")

    def load_processed_data(self):
        data_path = os.path.join(self.output_dir, "processed_data.pkl")
        logger.info(f"Đang load dữ liệu từ {data_path}")
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            self.vocab_dict = data["vocab"]
            self.vocab = list(
                self.vocab_dict.keys()
            )
            return data["texts"], data["labels"]
        except FileNotFoundError:
            logger.error(f"File {data_path} không tồn tại")
            raise


def main():

    DATA_PATH = "dataset/IMDB_Dataset.csv"
    VOCAB_SIZE = 15000
    MAX_LEN = 256
    OUTPUT_DIR = "dataset"

    preprocessor = IMDbPreprocessor(DATA_PATH, VOCAB_SIZE, MAX_LEN, OUTPUT_DIR)
    processed_data_path = os.path.join(OUTPUT_DIR, "processed_data.pkl")
    if not os.path.exists(processed_data_path):
        preprocessor.process_and_save()
    else:
        logger.info("Dữ liệu đã được xử lý trước đó, đang load...")

    texts, labels = preprocessor.load_processed_data()
    logger.info(f"Kích thước dữ liệu: {texts.shape} (texts), {labels.shape} (labels)")
    logger.info(f"Mẫu đầu tiên: texts[0][:10]={texts[0][:10]}, label[0]={labels[0]}")

if __name__ == "__main__":
    main()
