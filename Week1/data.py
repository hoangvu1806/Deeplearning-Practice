from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

print("Số lượng mẫu training:", len(train_dataset))
print("Số lượng mẫu test:", len(test_dataset))
print("Kích thước mẫu:", train_dataset[0][0].shape)