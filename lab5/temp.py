import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 步骤1: 载入并预处理数据
data = pd.read_csv('ionosphere.csv')     #从csv文件中读取数据集
X = data.iloc[:, :-1].values  # 使用‘iloc’函数将输入特征（X）和目标标签（y）分离。
y = data.iloc[:, -1].map({'g': 1, 'b': 0}).values  # 将类别标签转换为0和1

# 划分数据集为训练集和测试集，测试集占20%，random_state用于设置随机种子，确保可重复性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤2: 构建多层感知机模型
class MLP(nn.Module):  #定义一个MLP类，继承自'nn.Module'
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):  #模型有三层全连接层（线性层）：输入层、两个隐藏层和输出层。
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # 第二个隐藏层到输出层

    def forward(self, x): #使用ReLU激活函数作为隐藏层的激活函数，使用Sigmoid激活函数作为输出层的激活函数。
        x = F.relu(self.fc1(x))  # 输入层到第一个隐藏层的激活函数
        x = F.relu(self.fc2(x))  # 第一个隐藏层到第二个隐藏层的激活函数
        x = torch.sigmoid(self.fc3(x))  # 第二个隐藏层到输出层的激活函数
        return x

# 步骤3: 训练模型
input_dim = X_train.shape[1] #定义了输入维度
hidden_dim1 = 64    #隐藏层维度
hidden_dim2 = 32
output_dim = 1   #输出维度
batch_size = 18

model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)  #初始化模型
criterion = nn.BCELoss()  # 使用损失函数(二元交叉熵损失函数)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

def accuracy(y_true, y_pred):  #定义了一个用于计算准确率的辅助函数
    y_pred = (y_pred > 0.5).astype(np.int32)  # 阈值设为0.5进行二分类
    return np.mean(y_true == y_pred)

# 步骤4: 训练模型
train_accuracies = []  # 存储训练精度
test_accuracies = []   # 存储测试精度
epochs = 1000  #循环进行了1000个epochs的训练
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # 载入批处理数据
        batch_X = torch.Tensor(X_train[i:i + batch_size]).float()
        batch_y = torch.Tensor(y_train[i:i + batch_size]).float().view(-1, 1)

        # 初始化梯度
        optimizer.zero_grad()

        # 前向传播
        batch_outputs = model(batch_X)
        loss = criterion(batch_outputs, batch_y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 计算训练精度
    with torch.no_grad():
        train_inputs = torch.Tensor(X_train).float()
        train_outputs = model(train_inputs)
        train_predictions = (train_outputs > 0.5).numpy().flatten()
        train_accuracy = accuracy(y_train, train_predictions)
        train_accuracies.append(train_accuracy)

    # 计算测试精度
    with torch.no_grad():
        test_inputs = torch.Tensor(X_test).float()
        test_outputs = model(test_inputs)
        test_predictions = (test_outputs > 0.5).numpy().flatten()
        test_accuracy = accuracy(y_test, test_predictions)
        test_accuracies.append(test_accuracy)

    if (epoch + 1) % 100 == 0:  #每100个epochs打印一次损失值
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')

# 步骤5: 使用训练好的模型进行测试数据的前向传播预测
test_inputs = torch.Tensor(X_test).float()
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_predictions = (test_outputs > 0.5).numpy().flatten()


# 打印最终的训练精度和测试精度
final_train_accuracy = train_accuracies[-1]
final_test_accuracy = test_accuracies[-1]
print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')
