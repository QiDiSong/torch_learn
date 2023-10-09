import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成一些示例数据
# 这里我们假设 y = 2*x + 1，并加入一些噪声
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 * x + 1 + 0.1 * np.random.randn(100, 1)

# 将NumPy数组转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = LinearRegression()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率为0.01

# 训练模型
num_epochs = 1000  # 迭代次数
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印模型的参数
print(f'模型的参数：w = {model.linear.weight.item():.2f}, b = {model.linear.bias.item():.2f}')
