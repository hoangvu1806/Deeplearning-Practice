import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import json
from tqdm import tqdm
import wandb.sync
from utils import (
    load_cifar10,
    load_all_configs,
    visualize_feature_maps,
)
from models import CNN, ResNet20
import time


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
        epoch_start_time = time.time()

        # Huấn luyện
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_iter = tqdm(
            trainloader, desc=f"Epoch {epoch+1}/{epochs} - Training"
        )
        for inputs, labels in train_loader_iter:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)
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

        val_loader_iter = tqdm(valloader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in val_loader_iter:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
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

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{epochs}, Time: {epoch_time:.1f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
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
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
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


def plot_metrics(model_name, metrics, metric_type="accuracy"):
    plt.figure(figsize=(12, 5))

    for i, config_result in enumerate(metrics, 1):
        if metric_type == "accuracy":
            train_metric = config_result["history"]["train_acc"]
            val_metric = config_result["history"]["val_acc"]
            ylabel = "Accuracy (%)"
        else:  # loss
            train_metric = config_result["history"]["train_loss"]
            val_metric = config_result["history"]["val_loss"]
            ylabel = "Loss"

        plt.subplot(1, 3, i)
        plt.plot(train_metric, label="Train")
        plt.plot(val_metric, label="Validation")
        plt.title(f"Config {i}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()

    plt.suptitle(f"{model_name.upper()} {metric_type.capitalize()} Curves")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{model_name}_{metric_type}.png")
    plt.close()


def visualize_cnn_features(model, testloader, device, layer_idx=None):
    dataiter = iter(testloader)
    images, _ = next(dataiter)

    img = images[0].to(device)

    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((i, layer))

    print(f"Tìm thấy {len(conv_layers)} lớp tích chập trong mô hình CNN")

    # Trực quan hóa feature maps cho tất cả các lớp tích chập
    for idx, _ in conv_layers:
        print(f"Đang trực quan hóa lớp tích chập tại vị trí {idx}")
        visualize_feature_maps(model, img, f"layers.{idx}")
        plt.close() 


def get_conv_layers_info(model):
    """Lấy thông tin về tất cả các lớp tích chập trong mô hình"""
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((i, layer))

    print(f"Tổng số lớp tích chập: {len(conv_layers)}")
    print("\nChi tiết các lớp tích chập:")
    for i, (idx, layer) in enumerate(conv_layers, 1):
        print(f"Lớp tích chập thứ {i}:")
        print(f"  - Vị trí trong model.layers: {idx}")
        print(f"  - Kênh đầu vào (in_channels): {layer.in_channels}")
        print(f"  - Kênh đầu ra (out_channels): {layer.out_channels}")
        print(f"  - Kích thước kernel (kernel_size): {layer.kernel_size}")
        print(f"  - Padding: {layer.padding}")
        print(f"  - Stride: {layer.stride}")
        print()

    return conv_layers


def compare_results(cnn_results, resnet_results):
    # Tính toán trung bình và độ lệch chuẩn
    cnn_accs = [result["test_acc"] for result in cnn_results]
    resnet_accs = [result["test_acc"] for result in resnet_results]

    cnn_losses = [result["test_loss"] for result in cnn_results]
    resnet_losses = [result["test_loss"] for result in resnet_results]

    cnn_mean_acc = np.mean(cnn_accs)
    cnn_std_acc = np.std(cnn_accs)
    resnet_mean_acc = np.mean(resnet_accs)
    resnet_std_acc = np.std(resnet_accs)

    cnn_mean_loss = np.mean(cnn_losses)
    cnn_std_loss = np.std(cnn_losses)
    resnet_mean_loss = np.mean(resnet_losses)
    resnet_std_loss = np.std(resnet_losses)

    # Hiển thị kết quả
    print("\n=== So sánh kết quả ===")
    print(f"CNN - Độ chính xác trung bình: {cnn_mean_acc:.2f}% ± {cnn_std_acc:.2f}%")
    print(f"CNN - Loss trung bình: {cnn_mean_loss:.4f} ± {cnn_std_loss:.4f}")
    print(
        f"ResNet-20 - Độ chính xác trung bình: {resnet_mean_acc:.2f}% ± {resnet_std_acc:.2f}%"
    )
    print(
        f"ResNet-20 - Loss trung bình: {resnet_mean_loss:.4f} ± {resnet_std_loss:.4f}"
    )

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 5))

    # Biểu đồ độ chính xác
    plt.subplot(1, 2, 1)
    models = ["CNN", "ResNet-20"]
    accuracies = [cnn_mean_acc, resnet_mean_acc]
    errors = [cnn_std_acc, resnet_std_acc]

    bars = plt.bar(models, accuracies, yerr=errors, capsize=10)
    plt.ylabel("Test Accuracy (%)")
    plt.title("So sánh độ chính xác")

    # Thêm giá trị lên các cột
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracies[i]:.2f}%",
            ha="center",
            va="bottom",
        )

    # Biểu đồ loss
    plt.subplot(1, 2, 2)
    losses = [cnn_mean_loss, resnet_mean_loss]
    errors_loss = [cnn_std_loss, resnet_std_loss]

    bars = plt.bar(models, losses, yerr=errors_loss, capsize=10)
    plt.ylabel("Test Loss")
    plt.title("So sánh loss")

    # Thêm giá trị lên các cột
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{losses[i]:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("plots/comparison.png")
    plt.close()  # Đóng figure sau khi lưu

    # Lưu kết quả so sánh
    comparison = {
        "cnn": {
            "mean_acc": float(cnn_mean_acc),
            "std_acc": float(cnn_std_acc),
            "mean_loss": float(cnn_mean_loss),
            "std_loss": float(cnn_std_loss),
            "results": cnn_results,
        },
        "resnet": {
            "mean_acc": float(resnet_mean_acc),
            "std_acc": float(resnet_std_acc),
            "mean_loss": float(resnet_mean_loss),
            "std_loss": float(resnet_std_loss),
            "results": resnet_results,
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)


def run_experiment(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    all_configs = load_all_configs()
    configs = all_configs.get(model_name, [])

    results = []
    all_metrics = []

    max_batch_size = max(config.get("batch_size", 128) for config in configs)
    trainloader, valloader, testloader, classes = load_cifar10(
        batch_size=max_batch_size
    )

    for config in configs:
        config_num = config.get("config_num", 1)
        start_time = time.time()
        print(f"Đang huấn luyện {model_name} với cấu hình {config_num}: {config}")

        batch_size = config.get("batch_size", 128)
        if batch_size != max_batch_size:
            trainloader_config, valloader_config, testloader_config, _ = load_cifar10(
                batch_size=batch_size
            )
        else:
            trainloader_config, valloader_config, testloader_config = (
                trainloader,
                valloader,
                testloader,
            )

        if model_name == "cnn":
            model = CNN(
                num_classes=10,
                num_conv_layers=config.get("num_conv_layers", 3),
                base_filters=config.get("base_filters", 32),
            ).to(device)

            print("\n=== Thông tin về các lớp tích chập trong mô hình ===")
            get_conv_layers_info(model)

        elif model_name == "resnet":
            model = ResNet20(
                num_classes=10,
                base_filters=config.get("base_filters", 16),
            ).to(device)
        else:
            raise ValueError(f"Model {model_name} không được hỗ trợ")

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(
            config.get("optimizer"),
            model.parameters(),
            config.get("learning_rate"),
        )

        history, best_val_acc = train_model(
            model,
            trainloader_config,
            valloader_config,
            criterion,
            optimizer,
            device,
            config.get("epochs", 30),
            config,
            model_name,
        )
        model.load_state_dict(
            torch.load(f"models/{model_name}_config_{config_num}.pth")
        )
        test_loss, test_acc, predictions, true_labels = evaluate_model(
            model, testloader_config, criterion, device
        )

        # Tính tổng thời gian huấn luyện
        training_time = time.time() - start_time

        # Lưu kết quả
        result = {
            "config_num": config_num,
            "config": config,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time": training_time,
            "history": history,
        }
        results.append(result)
        all_metrics.append(result)

        print(f"Thời gian huấn luyện và đánh giá: {training_time:.2f} giây")
        if model_name == "cnn":
            visualize_cnn_features(model, testloader_config, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Vẽ biểu đồ metrics
    plot_metrics(model_name, all_metrics, "accuracy")
    plot_metrics(model_name, all_metrics, "loss")

    # Tính trung bình và độ lệch chuẩn các kết quả
    test_accs = [result["test_acc"] for result in results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)

    test_losses = [result["test_loss"] for result in results]
    mean_loss = np.mean(test_losses)
    std_loss = np.std(test_losses)

    print(f"\nKết quả cho {model_name}:")
    print(f"Độ chính xác trung bình trên tập test: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Loss trung bình trên tập test: {mean_loss:.4f} ± {std_loss:.4f}")

    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(
            {
                "results": results,
                "mean_acc": float(mean_acc),
                "std_acc": float(std_acc),
                "mean_loss": float(mean_loss),
                "std_loss": float(std_loss),
            },
            f,
            indent=4,
        )

    return results


def main():
    os.environ["WANDB_MODE"] = "online"
    num_threads = 8
    torch.set_num_threads(num_threads)
    print(f"Đặt số lượng CPU threads: {num_threads}")

    print("CIFAR-10 Image Classification - Huấn luyện và đánh giá các mô hình")

    print("=== Huấn luyện mô hình CNN cơ bản ===")
    cnn_results = run_experiment("cnn")

    print("\n=== Huấn luyện mô hình ResNet-20 ===")
    resnet_results = run_experiment("resnet")

    compare_results(cnn_results, resnet_results)

    print("\nHuấn luyện và đánh giá hoàn tất!")
    print("Kết quả được lưu trong thư mục 'results/'")
    print("Biểu đồ được lưu trong thư mục 'plots/'")
    print("Các model được lưu trong thư mục 'models/'")


if __name__ == "__main__":
    main()
