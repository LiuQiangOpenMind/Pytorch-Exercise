import torch.nn as nn
import torch as t
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# 注意事项：定义网络时，需要继承nn.Module，
# 并实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中。
# 如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，也可以不放，
# 但建议不放在其中，而在forward中使用nn.functional代替

# nn.Module是nn中最重要的类，
# 可把它看成是一个网络的封装，
# 包含网络各层定义以及forward方法，
# 调用forward(input)方法，
# 可返回前向传播的结果。

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        #nn.Module.__init__(self)

        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 15, 5)

        # 定义全连接层
        self.fc1 = nn.Linear(15*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # reshape, '-1'表示自适应???
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# 新建一个优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 先梯度清零
optimizer.zero_grad()

# 计算损失
input = t.randn(1, 1, 32, 32)
output = net(input)
target = t.arange(0, 10).view(1, 10)
target = target.float()     # 强制类型转换 Tensor后加long(), int(), double(), float(), byte()
criterion = nn.MSELoss()
loss = criterion(output, target)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()

print(net.conv1.bias.grad)

