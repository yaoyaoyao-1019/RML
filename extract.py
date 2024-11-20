import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据预处理的transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载CIFAR-10测试集
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# CIFAR-10标签列表
labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 创建一个字典，用于存储每个类别的索引和图像
class_images = {}

# 提取每个类别的一张图片
for data, target in testloader:
    label = labels[target.item()]
    if label not in class_images:
        class_images[label] = data.squeeze(0)  # 去除batch维度

# 显示提取的图片
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i, label in enumerate(labels):
    axes[i].imshow(class_images[label].permute(1, 2, 0))  # 调整通道顺序
    axes[i].set_title(label)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
