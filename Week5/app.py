import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from models import CNN, ResNet20

# Định nghĩa các lớp trong CIFAR-10
CLASSES = (
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

# Cấu hình mặc định cho các model
DEFAULT_CONFIGS = {
    "cnn": {
        1: {"num_conv_layers": 3, "base_filters": 32},
        2: {"num_conv_layers": 4, "base_filters": 64},
        3: {"num_conv_layers": 5, "base_filters": 128},
    },
    "resnet": {
        1: {"base_filters": 16},
        2: {"base_filters": 32},
        3: {"base_filters": 64},
    },
}


# Hàm tiền xử lý ảnh
def preprocess_image(image, model_type):
    # Chuyển đổi ảnh sang kích thước 32x32 (kích thước của ảnh CIFAR-10)
    image = image.resize((32, 32))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    return transform(image).unsqueeze(0)  # Thêm batch dimension


# Hàm dự đoán
def predict(model, image, device, model_type):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        probabilities = probabilities.cpu().numpy()

    return probabilities


# Tạo biểu đồ các xác suất dự đoán
def plot_prediction(probabilities):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Tạo DataFrame cho biểu đồ
    data = {"Class": list(CLASSES), "Probability": probabilities}

    # Sắp xếp theo xác suất giảm dần
    sorted_data = sorted(
        zip(data["Class"], data["Probability"]), key=lambda x: x[1], reverse=True
    )
    sorted_classes, sorted_probs = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(x=list(sorted_classes), y=list(sorted_probs), ax=ax)

    # Thêm giá trị lên các cột
    for i, p in enumerate(bars.patches):
        bars.annotate(
            f"{sorted_probs[i]:.2%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.title("Predicted Class Probabilities")
    plt.tight_layout()

    return fig


# Hàm tải model
def load_model(model_type, model_path, config_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sử dụng cấu hình mặc định
    config = DEFAULT_CONFIGS[model_type][config_num]

    if model_type == "cnn":
        model = CNN(
            num_classes=10,
            num_conv_layers=config.get("num_conv_layers", 3),
            base_filters=config.get("base_filters", 32),
        ).to(device)
    elif model_type == "resnet":
        model = ResNet20(
            num_classes=10, base_filters=config.get("base_filters", 16)
        ).to(device)
    else:
        raise ValueError(f"Model {model_type} không được hỗ trợ")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, device


def main():
    st.set_page_config(page_title="CIFAR-10 Image Classification", layout="wide")

    st.title("CIFAR-10 Image Classification")
    st.write("Ứng dụng demo phân loại ảnh sử dụng CNN và ResNet-20")

    # Chọn model
    model_options = {"CNN cơ bản": "cnn", "ResNet-20": "resnet"}

    model_choice = st.selectbox("Chọn mô hình:", list(model_options.keys()))
    model_type = model_options[model_choice]

    # Chọn config
    config_nums = [1, 2, 3]
    config_num = st.selectbox("Chọn cấu hình:", config_nums)

    # Đường dẫn đến model
    model_path = f"models/{model_type}_config_{config_num}.pth"

    # Kiểm tra xem model có tồn tại không
    if not os.path.exists(model_path):
        st.error(f"Model không tồn tại. Vui lòng huấn luyện mô hình trước khi sử dụng.")
        return

    # Tải model
    try:
        model, device = load_model(model_type, model_path, config_num)
        st.success(f"Đã tải mô hình {model_choice} thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        return

    # Upload image
    st.write("### Tải lên một ảnh để phân loại")
    uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Hiển thị ảnh đã tải lên
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Ảnh đã tải lên", width=300)

            # Tiền xử lý ảnh
            processed_image = preprocess_image(image, model_type)

            # Dự đoán
            probabilities = predict(model, processed_image, device, model_type)

            # Lấy nhãn dự đoán
            pred_class_idx = np.argmax(probabilities)
            pred_class = CLASSES[pred_class_idx]
            confidence = probabilities[pred_class_idx]

            # Hiển thị kết quả
            st.write(f"### Kết quả phân loại: {pred_class}")
            st.write(f"Độ tin cậy: {confidence:.2%}")

            # Hiển thị biểu đồ các xác suất
            fig = plot_prediction(probabilities)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    main()
