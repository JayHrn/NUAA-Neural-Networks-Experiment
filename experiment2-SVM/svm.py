import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #sklearn: 提供工具用于数据划分、标准化和评估。
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端（需要安装 tkinter）
import matplotlib.pyplot as plt
# 设置字体为 SimHei（黑体），支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # macOS 系统默认中文字体

plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
import torch #torch: PyTorch 是深度学习框架，用于定义模型、训练和评估。
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 数据来源: 威斯康星州乳腺癌数据集，包含肿瘤特征和诊断结果。
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None)


"""X: 数据的特征部分，从第 3 列开始到最后一列（前两列是 ID 和诊断标签）。
y: 目标变量，第二列为诊断结果，包含两类：
M: 恶性 (Malignant)
B: 良性 (Benign)"""
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

"""将目标变量 y 转换为数值形式：
1: 恶性 (M)
-1: 良性 (B)
np.where: 条件函数，满足 y == 'M' 的元素赋值为 1，否则为 -1。"""
y = np.where(y == 'M', 1, -1)
"""为了方便可视化，只选择前两个特征（radius_mean 和 texture_mean）。
原始数据集中有 30 个特征，但在二维平面中无法直接可视化。"""
X = X[:, :2]
"""train_test_split: 将数据划分为训练集和测试集。
test_size=0.2: 测试集占 20%。
random_state=42: 设置随机种子，保证结果可重复。
返回结果：
X_train 和 X_test: 训练集和测试集的特征。
y_train 和 y_test: 训练集和测试集的目标变量。
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""标准化: 对数据进行归一化处理，使每个特征具有零均值和单位方差。这对支持向量机 (SVM) 等模型至关重要，因为它们对特征的尺度敏感。"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""torch.tensor: 将 NumPy 数组转换为 PyTorch 张量。
dtype=torch.float32: 数据类型为 32 位浮点数。
view(-1, 1): 将目标变量 y 转换为二维张量，方便后续矩阵运算。
"""
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
"""TensorDataset: 将特征和目标变量封装为 PyTorch 数据集对象。
DataLoader: 将数据集分成小批次，用于训练：
batch_size=32: 每批数据包含 32 个样本。
shuffle=True: 在每个 epoch 之前对数据进行随机打乱，提高模型的泛化能力。"""
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

"""LinearSVM 类:
这是一个自定义的 PyTorch 模型，用于实现线性支持向量机 (SVM)。"""
"""__init__ 方法:

参数 input_dim: 输入特征的维度大小（即特征数量）。
self.fc: 定义了一个全连接层 (nn.Linear)。
input_dim: 输入特征数量。
1: 输出一个标量值，用于二分类任务。
bias=False: 不使用偏置项，因为线性 SVM 的分类超平面只需要权重，不需要偏置项。
forward 方法:

接收输入 x（特征向量），通过全连接层计算结果 self.fc(x)。
输出是一个标量值，表示输入样本到超平面的距离。"""

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1, bias=False)  # No bias term for SVM

    def forward(self, x):
        return self.fc(x)

"""HingeLoss 类:
自定义的 Hinge Loss 函数，用于训练支持向量机。
继承自 torch.nn.Module"""
class HingeLoss(nn.Module):
    def forward(self, outputs, targets):
        hinge_loss = torch.clamp(1 - outputs * targets, min=0)
        return torch.mean(hinge_loss)

"""input_dim: 输入特征的维度大小（等于 X_train 的列数，即所选的特征数量）。
model:
使用 LinearSVM 类初始化模型。
模型将接受特征向量作为输入，输出一个标量值"""
input_dim = X_train.shape[1]
model = LinearSVM(input_dim)

"""优化器 (optimizer):
使用随机梯度下降（SGD）优化器：
model.parameters(): 需要优化的模型参数（线性层的权重）。
lr=0.1: 学习率，控制每次参数更新的步长。
weight_decay=0.01: L2 正则化项，防止过拟合，相当于 SVM 中的正则化参数 
𝐶"""
criterion = HingeLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)  # L2 regularization as weight decay

"""开始实际的训练了"""
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        """outputs = model(X_batch): 使用线性 SVM 模型对输入 X_batch 进行预测。
loss = criterion(outputs, y_batch): 计算预测值与真实标签的 Hinge Loss"""
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

"""model.eval():
将模型设置为评估模式，关闭 dropout 和 batch normalization 等行为，确保预测稳定。
torch.no_grad():
禁用梯度计算，加快推理速度并节省显存。"""
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred = torch.sign(y_test_pred).view(-1).numpy()

print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

"""定义绘图的范围：
x_min, x_max 和 y_min, y_max: 确定特征 radius_mean 和 texture_mean 的值域范围，并扩展一定的边距。
生成网格点：
np.meshgrid: 在指定范围内生成二维网格，用于可视化决策边界。"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(scaler.transform(grid), dtype=torch.float32)

with torch.no_grad():
    Z = model(grid_tensor).view(-1).numpy()
    Z = Z.reshape(xx.shape)
"""绘制决策边界"""
# plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, colors=["#FFAAAA", "#AAAAFF", "#AAFFAA"])
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn, edgecolor="k")
# plt.title("良恶性乳腺癌 - 支持向量机")
# plt.xlabel("radius_mean")
# plt.ylabel("texture_mean")
# plt.show()

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, colors=["#FFAAAA", "#AAAAFF", "#AAFFAA"])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn, edgecolor="k")
plt.title("Breast Cancer SVM")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

# 保存图像到当前目录，文件名为 svm_breast_cancer.png，DPI 设置提高分辨率
plt.savefig("svm_breast_cancer.png", dpi=300)

# 显示图像（可选，如果你不想在屏幕上显示可删除此行）
# plt.show()
