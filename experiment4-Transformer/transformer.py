"""
# Transformer image classification with CutMix data augmentation on CIFAR-10 dataset
# Author: JayHrn
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Transformer 图像分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, img_size=32, patch_size=4, dim=256, depth=8, heads=8, mlp_dim=512):
        super(TransformerClassifier, self).__init__()
        self.patch_size = patch_size
        self.dim = dim

        self.embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout=0.2, batch_first=True),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.embedding(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, tokens, dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, tokens+1, dim)

        x += self.position_embeddings
        x = self.transformer(x)

        return self.mlp_head(x[:, 0])  # 只用 cls token 输出分类结果

# CutMix 数据增强
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = random.betavariate(alpha, alpha)
    bx1, by1, bx2, by2 = random_bbox(data.size(2), data.size(3), lam)
    data[:, :, bx1:bx2, by1:by2] = shuffled_data[:, :, bx1:bx2, by1:by2]

    return data, (targets, shuffled_targets, lam)


def random_bbox(height, width, lam):
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cy = random.randint(0, height)
    cx = random.randint(0, width)

    bx1 = max(0, cy - cut_h // 2)
    by1 = max(0, cx - cut_w // 2)
    bx2 = min(height, cy + cut_h // 2)
    by2 = min(width, cx + cut_w // 2)

    return bx1, by1, bx2, by2

if __name__ == "__main__":
    # 超参数
    batch_size = 128
    learning_rate = 0.0001
    epochs = 200
    num_classes = 10
    img_size = 32
    patch_size = 4

    # 数据预处理
    data_transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=data_transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=data_transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型初始化
    if torch.backends.mps.is_available():
        # 使用 Apple Silicon M3 的 Metal Performance Shaders
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = TransformerClassifier(num_classes, img_size, patch_size).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                              steps_per_epoch=len(train_loader), epochs=epochs)

    train_losses = []
    accuracies = []

    # === 训练过程 ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            if random.random() < 0.5:
                images, (labels_a, labels_b, lam) = cutmix(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # === 每个 epoch 后评估模型 ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    # === 保存模型权重 ===
    torch.save(model.state_dict(), "transformer_result_model.pth")

    # === 绘制 Loss 和 Accuracy 曲线 ===
    plt.figure(figsize=(12, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("transformer_result.png")