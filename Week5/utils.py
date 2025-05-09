import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def load_cifar10(batch_size=128, num_workers=4):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=False, transform=transform_train
    )

    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainset, [train_size, val_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, valloader, testloader, classes


def visualize_samples(dataloader, classes, num_samples=5):
    # Lấy một batch dữ liệu
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Hiển thị một số mẫu
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # Chuyển đổi tensor về dạng hình ảnh
        img = images[i].numpy().transpose((1, 2, 0))
        # Đảo ngược chuẩn hóa
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_all_configs():
    """Tải tất cả cấu hình từ file config.json"""
    with open("configs/config.json", "r") as f:
        return json.load(f)


def get_config(model_name, config_num):
    """Lấy cấu hình cụ thể từ file config.json"""
    configs = load_all_configs()
    if model_name not in configs:
        raise ValueError(f"Model {model_name} không tồn tại trong file cấu hình")

    for config in configs[model_name]:
        if config["config_num"] == config_num:
            return config

    raise ValueError(f"Không tìm thấy cấu hình số {config_num} cho model {model_name}")


def visualize_feature_maps(model, image, layer_name):
    # Hàm này sẽ được sử dụng để trực quan hóa feature maps từ CNN
    model.eval()

    # Thêm batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Hook để lấy feature map
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Đăng ký hook
    # Kiểm tra nếu layer_name có dạng "layers.X"
    if layer_name.startswith("layers.") and hasattr(model, "layers"):
        try:
            layer_idx = int(layer_name.split(".")[1])
            if 0 <= layer_idx < len(model.layers):
                model.layers[layer_idx].register_forward_hook(get_activation(layer_name))
            else:
                print(f"Chỉ số {layer_idx} nằm ngoài phạm vi của model.layers")
                return
        except (ValueError, IndexError):
            print(f"Layer {layer_name} không đúng định dạng (cần dạng layers.X)")
            return
    elif hasattr(model, layer_name):
        getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
    else:
        print(f"Layer {layer_name} không tồn tại trong mô hình")
        return

    # Forward pass
    with torch.no_grad():
        output = model(image)

    # Lấy feature map
    if layer_name not in activation:
        print(f"Không thể lấy được feature map từ {layer_name}")
        return
        
    feature_map = activation[layer_name][0].cpu()

    # Tạo thư mục lưu hình ảnh
    os.makedirs("plots/feature_maps", exist_ok=True)
    
    # Trực quan hóa
    num_features = min(64, feature_map.size(0))
    rows = int(np.sqrt(num_features))
    cols = int(np.ceil(num_features / rows))

    plt.figure(figsize=(12, 12))
    for i in range(num_features):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(f"Feature Maps từ layer {layer_name}")
    plt.tight_layout()
    
    # Lưu hình
    layer_id = layer_name.replace(".", "_")
    plt.savefig(f"plots/feature_maps/feature_map_{layer_id}.png")
    plt.close()
