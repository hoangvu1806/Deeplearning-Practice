import numpy as np
import pandas as pd
import os

from torchvision import datasets, transforms


# Tải dữ liệu MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./dataset", train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root="./dataset", train=False, download=False, transform=transform
)

# Chuyển đổi dữ liệu thành numpy arrays
train_data = []
train_labels = []
for images, labels in train_dataset:
    train_data.append(images.numpy().flatten())
    train_labels.append(labels)

test_data = []
test_labels = []
for images, labels in test_dataset:
    test_data.append(images.numpy().flatten())
    test_labels.append(labels)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


train_labels_one_hot = one_hot_encode(train_labels)
test_labels_one_hot = one_hot_encode(test_labels)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == "relu":
            self.a1 = self.relu(self.z1)
            self.da1 = self.relu_derivative(self.a1)
        elif self.activation == "sigmoid":
            self.a1 = self.sigmoid(self.z1)
            self.da1 = self.sigmoid_derivative(self.a1)
        elif self.activation == "tanh":
            self.a1 = self.tanh(self.z1)
            self.da1 = self.tanh_derivative(self.a1)

        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Tính gradient của layer 2
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Tính gradient của layer 1
        dz1 = np.dot(dz2, self.W2.T) * self.da1
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Cập nhật trọng số
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, batch_size, epochs, learning_rate, save_checkpoint=True):
        best_accuracy = 0
        for epoch in range(epochs):
            # trộn dữ liệu
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Training theo batch
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            if (epoch + 1) % 2 == 0:
                y_pred = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(y_pred + 1e-15), axis=1))
                accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                )

                if save_checkpoint and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if not os.path.exists("checkpoint"):
                        os.makedirs("checkpoint")
                    self.save_checkpoint("checkpoint/best_model.npy")
                    print(f"Đã lưu checkpoint với accuracy: {accuracy:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def save_checkpoint(self, filename):
        checkpoint = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation": self.activation,
        }
        np.save(filename, checkpoint)
        print(f"Đã lưu checkpoint vào {filename}")

    @classmethod
    def load_checkpoint(cls, filename):
        """Tải trọng số của mô hình từ file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint {filename}")

        checkpoint = np.load(filename, allow_pickle=True).item()
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            output_size=checkpoint["output_size"],
            activation=checkpoint["activation"],
        )

        model.W1 = checkpoint["W1"]
        model.b1 = checkpoint["b1"]
        model.W2 = checkpoint["W2"]
        model.b2 = checkpoint["b2"]

        print(f"Đã tải checkpoint từ {filename}")
        return model


def train_model():
    hyperparameters = [
        {
            "batch_size": 32,
            "learning_rate": 0.1,
            "hidden_size": 16,
            "activation": "relu",
        },
        {
            "batch_size": 16,
            "learning_rate": 0.01,
            "hidden_size": 64,
            "activation": "sigmoid",
        },
        {
            "batch_size": 64,
            "learning_rate": 0.05,
            "hidden_size": 32,
            "activation": "tanh",
        },
        {
            "batch_size": 128,
            "learning_rate": 0.001,
            "hidden_size": 128,
            "activation": "relu",
        },
        {
            "batch_size": 32,
            "learning_rate": 0.01,
            "hidden_size": 64,
            "activation": "sigmoid",
        },
    ]

    results = []

    for hp in hyperparameters:
        accuracies = []
        for run in range(5):
            nn = NeuralNetwork(
                input_size=784,
                hidden_size=hp["hidden_size"],
                output_size=10,
                activation=hp["activation"],
            )

            nn.train(
                train_data,
                train_labels_one_hot,
                batch_size=hp["batch_size"],
                epochs=10,
                learning_rate=hp["learning_rate"],
                save_checkpoint=(run == 0),
            )  # Chỉ lưu checkpoint cho lần chạy đầu tiên

            y_pred = nn.predict(test_data)
            acc = np.mean(y_pred == test_labels)
            accuracies.append(acc)
            print(f"Siêu tham số: {hp}, Lần chạy {run+1}, Độ chính xác: {acc:.4f}")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results.append(
            {"hyperparameters": hp, "mean_accuracy": mean_acc, "std_accuracy": std_acc}
        )
        print(
            f"Siêu tham số: {hp}, Độ chính xác trung bình: {mean_acc:.4f}, Độ lệch chuẩn: {std_acc:.4f}"
        )

    # Lưu kết quả vào file CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Kết quả đã được lưu vào file results.csv")


if __name__ == "__main__":
    train_model()
