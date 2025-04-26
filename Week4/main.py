import os
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import time
import copy
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import seaborn as sns

from utils.dataset import (
    load_processed_vocab,
    create_processed_data_loaders,
    Vocabulary,
)
from utils.metrics import evaluate_predictions
from models.lstm_attention import LSTMAttention
import config

# Số lượng epochs cố định
EPOCHS = 10

# Định nghĩa các cấu hình siêu tham số
CONFIGURATIONS = [
    {
        "name": "config_1",
        "description": "Baseline",
        "params": {
            "hidden_dim": 128,
            "attention_dim": 64,
            "embedding_dim": 300,
            "num_layers": 2,
            "batch_size": 32,
            "dropout": 0.5,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
    },
    {
        "name": "config_2",
        "description": "Larger model",
        "params": {
            "hidden_dim": 256,
            "attention_dim": 128,
            "embedding_dim": 512,
            "num_layers": 3,
            "batch_size": 32,
            "dropout": 0.4,
            "learning_rate": 0.001,
            "optimizer": "adamw",
        },
    },
    {
        "name": "config_3",
        "description": "Different learning rate and optimizer",
        "params": {
            "hidden_dim": 128,
            "attention_dim": 64,
            "embedding_dim": 300,
            "num_layers": 2,
            "batch_size": 16,
            "dropout": 0.3,
            "learning_rate": 0.003,
            "optimizer": "rmsprop",
        },
    },
    {
        "name": "config_4",
        "description": "Small batches with high learning rate",
        "params": {
            "hidden_dim": 128,
            "attention_dim": 64,
            "embedding_dim": 300,
            "num_layers": 2,
            "batch_size": 8,
            "dropout": 0.3,
            "learning_rate": 0.01,
            "optimizer": "adam",
        },
    },
    {
        "name": "config_5",
        "description": "Fewer layers with larger embeddings",
        "params": {
            "hidden_dim": 128,
            "attention_dim": 128,
            "embedding_dim": 512,
            "num_layers": 2,
            "batch_size": 32,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "optimizer": "nadam",
        },
    },
]


def set_seed(seed: int) -> None:
    """Cố định các seed để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(
    model: nn.Module, optimizer_name: str, learning_rate: float
) -> optim.Optimizer:
    """Tạo optimizer theo tên."""
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "nadam":
        return optim.NAdam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} không được hỗ trợ")


def train_model_with_config(
    config_name: str, config_params: Dict[str, Any], run_id: int
) -> Dict[str, Any]:
    """Huấn luyện mô hình với một cấu hình và ID chạy cụ thể, log kết quả lên Wandb."""

    run_name = f"{config_name}_run[{run_id}]"
    wandb.init(
        project="vietnamese-news-classification",
        name=run_name,
        config=config_params,
        reinit=True,
    )

    set_seed(42 + run_id)

    processed_dir = "processed_data"
    vocab_path = os.path.join(processed_dir, "vocab.pkl")

    print(f"Đang tải dữ liệu đã xử lý cho {run_name}...")
    vocab = load_processed_vocab(vocab_path)

    train_loader, val_loader, test_loader = create_processed_data_loaders(
        processed_dir, vocab, config.MAX_SEQ_LEN, config_params["batch_size"]
    )

    model = LSTMAttention(
        vocab_size=vocab.vocab_size,
        embedding_dim=config_params["embedding_dim"],
        hidden_dim=config_params["hidden_dim"],
        attention_dim=config_params["attention_dim"],
        output_dim=len(config.LABEL_MAP),
        num_layers=config_params["num_layers"],
        bidirectional=config.BIDIRECTIONAL,
        dropout=config_params["dropout"],
        pad_idx=vocab.pad_idx,
    )

    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model, config_params["optimizer"], config_params["learning_rate"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    experiment_dir = f"experiments/{config_name}/run_{run_id}"
    os.makedirs(experiment_dir, exist_ok=True)
    model_save_path = os.path.join(experiment_dir, "best_model.pt")

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    lr_history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    print(f"Bắt đầu huấn luyện cho {run_name}...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()
        epoch_loss = 0
        true_labels = []
        pred_labels = []

        for batch in train_loader:

            text, labels = batch
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions, _ = model(text)

            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * text.size(0)

            _, preds = torch.max(predictions, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        train_metric = evaluate_predictions(true_labels, pred_labels)
        train_metrics.append(train_metric)

        model.eval()
        val_loss = 0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text, labels = batch
                text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)

                predictions, _ = model(text)
                loss = criterion(predictions, labels)

                val_loss += loss.item() * text.size(0)

                _, preds = torch.max(predictions, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        val_metric = evaluate_predictions(true_labels, pred_labels)
        val_metrics.append(val_metric)

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        scheduler.step()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metric["accuracy"],
                "val_accuracy": val_metric["accuracy"],
                "train_f1": train_metric["f1_macro"],
                "val_f1": val_metric["f1_macro"],
                "learning_rate": current_lr,
            }
        )

        print(f"Epoch: {epoch}/{EPOCHS}")
        print(
            f'Train Loss: {epoch_loss:.4f}, Accuracy: {train_metric["accuracy"]:.4f}, F1: {train_metric["f1_macro"]:.4f}'
        )
        print(
            f'Val Loss: {val_loss:.4f}, Accuracy: {val_metric["accuracy"]:.4f}, F1: {val_metric["f1_macro"]:.4f}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        wandb.log({"gradient_norm": total_norm})

    training_time = time.time() - start_time
    print(f"Hoàn thành huấn luyện cho {run_name} trong {training_time/60:.2f} phút")
    print(f"Epoch tốt nhất: {best_epoch} với validation loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_loss = 0
    true_labels = []
    pred_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            text, labels = batch
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE)

            predictions, _ = model(text)
            loss = criterion(predictions, labels)

            test_loss += loss.item() * text.size(0)

            _, preds = torch.max(predictions, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_metrics = evaluate_predictions(true_labels, pred_labels)

    wandb.log(
        {
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
            "training_time_minutes": training_time / 60,
            "best_epoch": best_epoch,
        }
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log(
        {"total_parameters": total_params, "trainable_parameters": trainable_params}
    )

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": [m["accuracy"] for m in train_metrics],
        "val_accuracy": [m["accuracy"] for m in val_metrics],
        "train_f1": [m["f1_macro"] for m in train_metrics],
        "val_f1": [m["f1_macro"] for m in val_metrics],
        "learning_rate": lr_history,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "best_epoch": best_epoch,
    }

    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.axhline(y=test_loss, color="r", linestyle="-", label="Test Loss")
    plt.axvline(
        x=best_epoch - 1, color="g", linestyle="--", label=f"Best Epoch ({best_epoch})"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot([m["accuracy"] for m in train_metrics], label="Train Accuracy")
    plt.plot([m["accuracy"] for m in val_metrics], label="Val Accuracy")
    plt.axhline(
        y=test_metrics["accuracy"], color="r", linestyle="-", label="Test Accuracy"
    )
    plt.axvline(
        x=best_epoch - 1, color="g", linestyle="--", label=f"Best Epoch ({best_epoch})"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(lr_history, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "training_curves.png"))
    plt.close()

    wandb.finish()

    results = {
        "config_name": config_name,
        "run_id": run_id,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "training_time": training_time / 60,
        "best_epoch": best_epoch,
    }

    return results


def plot_comparison_results(summary_df: pd.DataFrame, results_dir: str):
    """Vẽ và lưu biểu đồ so sánh kết quả giữa các cấu hình."""

    config_names = summary_df["config_name"]
    avg_accuracy = summary_df["avg_test_accuracy"]
    std_accuracy = summary_df["std_test_accuracy"]
    avg_f1 = summary_df["avg_test_f1"]
    std_f1 = summary_df["std_test_f1"]
    avg_error = summary_df["avg_test_error"]
    std_error = summary_df["std_test_error"]

    x = np.arange(len(config_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x, avg_accuracy, width, yerr=std_accuracy, label="Avg Accuracy", capsize=5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Average Test Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        x, avg_f1, width, yerr=std_f1, label="Avg F1 Macro", capsize=5, color="orange"
    )
    ax.set_ylabel("F1 Macro Score")
    ax.set_title("Average Test F1 Macro Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_comparison.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        x,
        avg_error,
        width,
        yerr=std_error,
        label="Avg Error Rate",
        capsize=5,
        color="red",
    )
    ax.set_ylabel("Error Rate (1 - Accuracy)")
    ax.set_title("Average Test Error Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_comparison.png"))
    plt.close(fig)

    print(f"\nBiểu đồ so sánh đã được lưu tại thư mục: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá mô hình với nhiều cấu hình"
    )
    parser.add_argument(
        "--configs", nargs="+", type=str, help="Tên các cấu hình muốn chạy"
    )
    parser.add_argument("--all", action="store_true", help="Chạy tất cả các cấu hình")
    parser.add_argument(
        "--runs", type=int, default=3, help="Số lần chạy cho mỗi cấu hình"
    )
    args = parser.parse_args()

    if args.all:
        configs_to_run = CONFIGURATIONS
    elif args.configs:
        configs_to_run = [
            config for config in CONFIGURATIONS if config["name"] in args.configs
        ]
    else:
        print("Vui lòng chỉ định cấu hình để chạy với --configs hoặc --all")
        return

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    summary_results = []

    for (
        config_item
    ) in configs_to_run:  # Đổi tên biến để tránh nhầm lẫn với module config
        config_name = config_item["name"]
        config_params = config_item["params"]
        config_results = []

        print(f"\n{'=' * 50}")
        print(f"Bắt đầu huấn luyện với cấu hình: {config_name}")
        print(f"Mô tả: {config_item['description']}")  # Sử dụng config_item
        print(f"Tham số: {config_params}")
        print(f"{'=' * 50}\n")

        for run_id in range(1, args.runs + 1):
            print(f"\nChạy lần {run_id}/{args.runs} cho cấu hình {config_name}")
            results = train_model_with_config(config_name, config_params, run_id)
            config_results.append(results)
            all_results.append(results)

        test_losses = [r["test_loss"] for r in config_results]
        test_accuracies = [r["test_accuracy"] for r in config_results]
        test_f1s = [r["test_f1"] for r in config_results]
        test_errors = [1 - acc for acc in test_accuracies]

        avg_test_loss = np.mean(test_losses)
        std_test_loss = np.std(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)
        std_test_accuracy = np.std(test_accuracies)
        avg_test_f1 = np.mean(test_f1s)
        std_test_f1 = np.std(test_f1s)
        avg_test_error = np.mean(test_errors)
        std_test_error = np.std(test_errors)

        summary = {
            "config_name": config_name,
            "description": config_item["description"],  # Sử dụng config_item
            "avg_test_loss": avg_test_loss,
            "std_test_loss": std_test_loss,
            "avg_test_accuracy": avg_test_accuracy,
            "std_test_accuracy": std_test_accuracy,
            "avg_test_f1": avg_test_f1,
            "std_test_f1": std_test_f1,
            "avg_test_error": avg_test_error,
            "std_test_error": std_test_error,
            "params": config_params,
        }
        summary_results.append(summary)

        print(f"\n{'=' * 50}")
        print(f"Kết quả tổng hợp cho cấu hình {config_name}:")
        print(f"Mô tả: {config_item['description']}")  # Sử dụng config_item
        print(f"Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
        print(f"Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        print(f"Test Error: {avg_test_error:.4f} ± {std_test_error:.4f}")
        print(f"Test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"{'=' * 50}\n")

    detailed_results_path = os.path.join(results_dir, "detailed_results.json")
    with open(detailed_results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    summary_results_path = os.path.join(results_dir, "summary_results.json")
    with open(summary_results_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df[
        [
            "config_name",
            "description",
            "avg_test_loss",
            "std_test_loss",
            "avg_test_accuracy",
            "std_test_accuracy",
            "avg_test_error",
            "std_test_error",
            "avg_test_f1",
            "std_test_f1",
        ]
    ]

    print("\nKết quả tổng hợp cho tất cả các cấu hình:")
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(results_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    plot_comparison_results(summary_df, results_dir)

    best_config_idx = summary_df["avg_test_accuracy"].idxmax()
    best_config = summary_df.iloc[best_config_idx]

    print(
        f"\nCấu hình tốt nhất (dựa trên Accuracy cao nhất): {best_config['config_name']}"
    )
    print(f"Mô tả: {best_config['description']}")
    print(
        f"Test Accuracy: {best_config['avg_test_accuracy']:.4f} ± {best_config['std_test_accuracy']:.4f}"
    )
    print(
        f"Test Error: {best_config['avg_test_error']:.4f} ± {best_config['std_test_error']:.4f}"
    )
    print(
        f"Test F1: {best_config['avg_test_f1']:.4f} ± {best_config['std_test_f1']:.4f}"
    )

    print(f"\nKết quả chi tiết đã được lưu tại: {detailed_results_path}")
    print(f"Kết quả tổng hợp đã được lưu tại: {summary_results_path}")
    print(f"Bảng kết quả tổng hợp đã được lưu tại: {summary_csv_path}")


if __name__ == "__main__":
    main()
