import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SIZE = 25000
TEST_SIZE = 5000


def train(
    model_type,
    vocab_size,
    embedding_dim,
    hidden_size,
    num_layers,
    output_size,
    batch_size,
    learning_rate,
    optimizer_type,
    epochs,
    data_path,
    device="cpu",
):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    texts = data["texts"]
    labels = data["labels"]
    vocab_dict = data["vocab"]
    pad_idx = vocab_dict["<pad>"]
    train_texts, test_texts = (
        texts[:TRAIN_SIZE],
        texts[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE],
    )
    train_labels, test_labels = (
        labels[:TRAIN_SIZE],
        labels[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE],
    )

    train_dataset = TensorDataset(train_texts, train_labels)
    test_dataset = TensorDataset(test_texts, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "RNN":
        from Week2.models import RNN

        model = RNN(
            vocab_size, embedding_dim, hidden_size, num_layers, output_size, pad_idx
        )
    elif model_type == "LSTM":
        from Week2.models import LSTM

        model = LSTM(
            vocab_size, embedding_dim, hidden_size, num_layers, output_size, pad_idx
        )
    else:
        raise ValueError("model_type phải là 'RNN' hoặc 'LSTM'")
    model.to(device)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            "optimizer_type phải là 'adam', 'adamw', 'rmsprop', hoặc 'nadam'"
        )

    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(
                    device
                )
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = correct / total
        test_accuracies.append(accuracy)

        logger.info(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}"
        )

    return model, train_losses, test_accuracies, test_losses


def run_experiments(configs, runs=3):
    results = []
    # Theo dõi test loss tốt nhất cho RNN và LSTM trên tất cả các runs/configs
    global_best_rnn_loss = float("inf")
    global_best_lstm_loss = float("inf")
    global_best_rnn_state = None
    global_best_lstm_state = None
    global_best_rnn_config = None
    global_best_lstm_config = None

    for config in configs:
        config_results = {"config": config, "accuracies": [], "test_losses": []}
        logger.info(f"Chạy cấu hình: {config}")
        for run in range(runs):
            logger.info(f"Run {run+1}/{runs}")
            model, train_losses, test_accuracies, test_losses = train(**config)
            config_results["accuracies"].append(test_accuracies[-1])
            config_results["test_losses"].append(test_losses[-1])

            if config["model_type"] == "RNN" and test_losses[-1] < global_best_rnn_loss:
                global_best_rnn_loss = test_losses[-1]
                global_best_rnn_state = model.state_dict()
                global_best_rnn_config = config
                torch.save(
                    global_best_rnn_state,
                    f"checkpoint/best_RNN_{config['num_layers']}_{config['hidden_size']}_{config['optimizer_type']}.pth",
                )
                logger.info(
                    f"Đã lưu checkpoint toàn cục RNN tại checkpoint/best_RNN_{config['num_layers']}_{config['hidden_size']}_{config['optimizer_type']}.pth, Test Loss: {global_best_rnn_loss:.4f}"
                )
            elif (
                config["model_type"] == "LSTM"
                and test_losses[-1] < global_best_lstm_loss
            ):
                global_best_lstm_loss = test_losses[-1]
                global_best_lstm_state = model.state_dict()
                global_best_lstm_config = config
                torch.save(
                    global_best_lstm_state,
                    f"checkpoint/best_LSTM_{config['num_layers']}_{config['hidden_size']}_{config['optimizer_type']}.pth",
                )
                logger.info(
                    f"Đã lưu checkpoint toàn cục LSTM tại checkpoint/best_LSTM_{config['num_layers']}_{config['hidden_size']}_{config['optimizer_type']}.pth, Test Loss: {global_best_lstm_loss:.4f}"
                )

        mean_acc = np.mean(config_results["accuracies"])
        std_acc = np.std(config_results["accuracies"])
        mean_test_loss = np.mean(config_results["test_losses"])
        std_test_loss = np.std(config_results["test_losses"])
        config_results["mean_acc"] = mean_acc
        config_results["std_acc"] = std_acc
        config_results["mean_test_loss"] = mean_test_loss
        config_results["std_test_loss"] = std_test_loss
        results.append(config_results)
        logger.info(
            f"Cấu hình: {config['model_type']}, Mean Acc: {mean_acc:.4f}, Std Acc: {std_acc:.4f}, "
            f"Mean Test Loss: {mean_test_loss:.4f}, Std Test Loss: {std_test_loss:.4f}"
        )

    if global_best_rnn_state is not None:
        logger.info(
            f"Checkpoint tốt nhất RNN: Config: {global_best_rnn_config}, Test Loss: {global_best_rnn_loss:.4f}"
        )
    if global_best_lstm_state is not None:
        logger.info(
            f"Checkpoint tốt nhất LSTM: Config: {global_best_lstm_config}, Test Loss: {global_best_lstm_loss:.4f}"
        )

    return results


def main():
    configs = [
        {
            "model_type": "LSTM",
            "vocab_size": 15000,
            "embedding_dim": 128,
            "hidden_size": 64,
            "num_layers": 1,
            "output_size": 2,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "epochs": 10,
            "data_path": "dataset/processed_data.pkl",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        {
            "model_type": "RNN",
            "vocab_size": 15000,
            "embedding_dim": 128,
            "hidden_size": 128,
            "num_layers": 1,
            "output_size": 2,
            "batch_size": 32,
            "learning_rate": 0.003,
            "optimizer_type": "rmsprop",
            "epochs": 10,
            "data_path": "dataset/processed_data.pkl",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        {
            "model_type": "LSTM",
            "vocab_size": 15000,
            "embedding_dim": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 2,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer_type": "adamw",
            "epochs": 10,
            "data_path": "dataset/processed_data.pkl",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        {
            "model_type": "RNN",
            "vocab_size": 15000,
            "embedding_dim": 128,
            "hidden_size": 32,
            "num_layers": 4,
            "output_size": 2,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer_type": "adamw",
            "epochs": 10,
            "data_path": "dataset/processed_data.pkl",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        {
            "model_type": "LSTM",
            "vocab_size": 15000,
            "embedding_dim": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 2,
            "batch_size": 128,
            "learning_rate": 0.002,
            "optimizer_type": "nadam",
            "epochs": 10,
            "data_path": "dataset/processed_data.pkl",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    ]

    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    results = run_experiments(configs, runs=3)

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "config": res["config"],
                    "accuracies": res["accuracies"],
                    "mean_acc": res["mean_acc"],
                    "std_acc": res["std_acc"],
                }
                for res in results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Đã lưu kết quả vào 'results/results.json'")


if __name__ == "__main__":
    main()
