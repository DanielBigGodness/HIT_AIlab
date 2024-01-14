import torch  # 导入PyTorch库
from torchvision import transforms  # 导入图像处理工具
from torch.utils.data import DataLoader  # 导入数据加载器
from torchvision import datasets  # 导入数据集
import torch.nn.functional as F  # 导入函数库，例如relu()
import torch.optim as optim  # 导入优化器库
import torch  # 再次导入PyTorch库，这行代码似乎是多余的
import torch.nn as nn  # 导入神经网络库

# 准备数据集(dataset和dataloader)

batch_size = 64  # 设置批处理大小
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 定义图像变换操作，包括将图像转为Tensor和归一化处理
train_dataset = datasets.MNIST(root='../dataset/minist/', train=True, download=True, transform=transform)  # 定义训练数据集，使用定义好的transform进行处理

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  # 定义训练数据加载器

test_dataset = datasets.MNIST(root='../dataset/minist/', train=False, download=True, transform=transform)  # 定义测试数据集

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)  # 定义测试数据加载器

class Net(torch.nn.Module):  # 定义神经网络类，继承自torch.nn.Module
    def __init__(self):  # 初始化函数
        super(Net, self).__init__()  # 调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)  # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)  # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 定义第一个全连接层，输入节点数为64*5*5，输出节点数为128
        self.fc2 = nn.Linear(128, 10)  # 定义第二个全连接层，输入节点数为128，输出节点数为10

    #inputs: 64 x 2304 = batch x 2304
    # mat1 = 64 *  1600
    # mat2 = 64 x 6 x 6   *     128

    def forward(self, x):  # 前向传播函数
        x = nn.functional.relu(self.conv1(x))  # 对输入x进行第一次卷积操作并通过relu激活函数
        x = nn.functional.max_pool2d(x, 2)  # 对x进行最大池化操作，池化窗口大小为2
        x = nn.functional.relu(self.conv2(x))  # 对x进行第二次卷积操作并通过relu激活函数
        x = nn.functional.max_pool2d(x, 2)  # 对x进行最大池化操作，池化窗口大小为2
        x = x.view(x.size(0), -1)  # 将x展平为一维向量，准备进行全连接操作
        x = nn.functional.relu(self.fc1(x))  # 对x进行第一次全连接操作并通过relu激活函数
        x = self.fc2(x)  # 对x进行第二次全连接操作
        return x

model = Net()   # 实例化神经网络类

criterion = torch.nn.CrossEntropyLoss()   # 使用交叉熵损失函数。注意：神经网络最后一层无需做激活，因为将其变分布的softmax激活包含在交叉熵损失中。
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)   # 使用随机梯度下降优化器，并设置学习率和动量。

# 训练循环
def train(epoch):   # 定义训练函数，参数为训练轮数。
    running_loss = 0.0   # 初始化累计损失
    for batch_idx, data in enumerate(train_loader, 0):   # 对训练数据进行遍历
        inputs, target = data   # 获取输入数据和目标标签

        optimizer.zero_grad()   # 在优化器优化之前要清零梯度

        outputs = model(inputs)   # 对输入数据进行前向传播，得到输出结果
        loss = criterion(outputs, target)   # 计算损失
        loss.backward()   # 反向传播，计算梯度
        optimizer.step()   # 更新权重

        running_loss += loss.item()   # 累计损失

        if batch_idx % 100 == 99:  # 每300个批次输出一次损失
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

def test():   # 定义测试函数
    correct = 0   # 初始化正确预测的样本数
    total = 0   # 初始化总样本数
    with torch.no_grad():  # 测试过程中不需要计算梯度
        for data in test_loader:   # 对测试数据进行遍历
            images, labels = data   # 获取输入数据和目标标签
            outputs = model(images)   # 对输入数据进行前向传播，得到输出结果
            _, predicted = torch.max(outputs.data, dim=1)   # 获取每行最大值的下标，即预测结果
            total += labels.size(0)   # 累计样本数
            correct += (predicted == labels).sum().item()   # 累计正确预测的样本数
    print('Accuracy on testset: %.6f %%' % (100 * correct / total))   # 输出测试集上的准确率

    with torch.no_grad():  # 测试过程中不需要计算梯度
        for data in train_loader:   # 对训练数据进行遍历
            images, labels = data   # 获取输入数据和目标标签
            outputs = model(images)   # 对输入数据进行前向传播，得到输出结果
            _, predicted = torch.max(outputs.data, dim=1)   # 获取每行最大值的下标，即预测结果
            total += labels.size(0)   # 累计样本数
            correct += (predicted == labels).sum().item()   # 累计正确预测的样本数
    print('Accuracy on trainset: %.6f  %%' % (100 * correct / total))  # 输出训练集上的准确率


if __name__ == '__main__':
    for epoch in range(20):  # 训练20轮，每轮都进行训练和测试。
        train(epoch)
        test()