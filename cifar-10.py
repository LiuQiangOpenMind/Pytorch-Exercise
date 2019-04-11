import torch as t
import torchvision as tv
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToPILImage
show = ToPILImage() # 将Tensor转成Image, 方便可视化

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(), # 转化为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='./data/',
    train=True,
    download=True,
    transform=transform
)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size = 4,
    shuffle = True,
    num_workers = 0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# (data, label) = trainset[100]
# print(classes[label])
#
# show((data+1)/2).resize((100, 100))

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        #nn.Module.__init__(self)

        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
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

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络

t.set_num_threads(8)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

    print('Finished Training')


