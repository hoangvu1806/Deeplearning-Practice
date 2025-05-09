import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    load_cifar10,
    get_config,
    visualize_feature_maps,
)
from models import CNN, ResNet20


def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model_parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} không được hỗ trợ")


def train_model(
    model,
    trainloader,
    valloader,
    criterion,
    optimizer,
    device,
    epochs,
    config,
    model_name,
):
    wandb.init(
        project="cifar10-classification",
        config=config,
        name=f"{model_name}_config_{config.get('config_num', 1)}",
    )

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            trainloader, desc=f"Epoch {epoch+1}/{epochs} - Training"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                valloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(
                model.state_dict(),
                f'models/{model_name}_config_{config.get("config_num", 1)}.pth',
            )
            print(f"Đã lưu model tại epoch {epoch+1} với val_acc = {val_acc:.2f}%")

    wandb.finish()

    # Trả về lịch sử huấn luyện
    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    }

    return history, best_val_acc


def evaluate_model(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(testloader)
    test_acc = 100.0 * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return test_loss, test_acc, np.array(all_preds), np.array(all_labels)


def run_experiment(model_name, config_nums=None):
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    if config_nums is None:
        config_nums = [1, 2, 3]

    results = []

    for config_num in config_nums:
        # Tải cấu hình siêu tham số từ file cấu hình duy nhất
        config = get_config(model_name, config_num)
        print(f"Đang huấn luyện {model_name} với cấu hình {config_num}: {config}")

        # Tải dữ liệu
        batch_size = config.get("batch_size", 128)
        trainloader, valloader, testloader, classes = load_cifar10(
            batch_size=batch_size
        )

        # Khởi tạo model
        if model_name == "cnn":
            model = CNN(
                num_classes=10,
                num_conv_layers=config.get("num_conv_layers", 3),
                base_filters=config.get("base_filters", 32),
            ).to(device)
        elif model_name == "resnet":
            model = ResNet20(
                num_classes=10,
                base_filters=config.get("base_filters", 16),
            ).to(device)
        else:
            raise ValueError(f"Model {model_name} không được hỗ trợ")

        # Khởi tạo loss function và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(
            config.get("optimizer", "adam"),
            model.parameters(),
            config.get("learning_rate", 0.001),
        )

        # Huấn luyện model
        history, best_val_acc = train_model(
            model,
            trainloader,
            valloader,
            criterion,
            optimizer,
            device,
            config.get("epochs", 30),
            config,
            model_name,
        )

        # Đánh giá model trên tập test
        model.load_state_dict(
            torch.load(f"models/{model_name}_config_{config_num}.pth")
        )
        test_loss, test_acc, predictions, true_labels = evaluate_model(
            model, testloader, criterion, device
        )

        # Lưu kết quả
        result = {
            "config_num": config_num,
            "config": config,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        results.append(result)

        # Nếu là model CNN, trực quan hóa feature maps
        if model_name == "cnn":
            # Lấy một batch dữ liệu
            dataiter = iter(testloader)
            images, _ = next(dataiter)

            # Chọn một ảnh để trực quan hóa
            img = images[0].to(device)

            # Lấy tên của layer cuối cùng trong model.layers
            last_conv_idx = 0
            for i, layer in enumerate(model.layers):
                if isinstance(layer, nn.Conv2d):
                    last_conv_idx = i

            # Trực quan hóa feature maps
            visualize_feature_maps(model, img, f"layers.{last_conv_idx}")

    # Tính trung bình và độ lệch chuẩn các kết quả
    test_accs = [result["test_acc"] for result in results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)

    print(f"\nKết quả cho {model_name}:")
    print(f"Độ chính xác trung bình trên tập test: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(
            {"results": results, "mean_acc": mean_acc, "std_acc": std_acc}, f, indent=4
        )

    return results, mean_acc, std_acc


if __name__ == "__main__":
    # Huấn luyện và đánh giá model CNN
    print("=== Huấn luyện mô hình CNN cơ bản ===")
    cnn_results, cnn_mean_acc, cnn_std_acc = run_experiment("cnn")

    # Huấn luyện và đánh giá model ResNet20
    print("\n=== Huấn luyện mô hình ResNet20 ===")
    resnet_results, resnet_mean_acc, resnet_std_acc = run_experiment("resnet")
