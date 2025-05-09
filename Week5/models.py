import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, num_conv_layers=3, base_filters=32):
        super(CNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Conv2d(3, base_filters, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(base_filters))
        self.layers.append(nn.ReLU())
        
        # Thêm các lớp tích chập theo yêu cầu
        for i in range(1, num_conv_layers):
            in_channels = base_filters * (2**(i-1))
            out_channels = base_filters * (2**i)
            
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
        
        # Tính toán kích thước đầu ra sau các lớp tích chập
        output_size = 32 // (2**(num_conv_layers-1))
        final_filters = base_filters * (2**(num_conv_layers-1))
        
        # Lớp fully connected
        self.fc = nn.Linear(final_filters * output_size * output_size, num_classes)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_filters=16):
        super(ResNet, self).__init__()
        self.in_planes = base_filters

        self.conv1 = nn.Conv2d(3, base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)

        self.layer1 = self._make_layer(block, base_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_filters*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_filters*4, num_blocks[2], stride=2)

        self.linear = nn.Linear(base_filters*4*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(num_classes=10, base_filters=16):
    """ResNet-20 cho CIFAR-10"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, base_filters=base_filters) 