import torch as t
import numpy as np

# 构建5*3 矩阵， 只是分配了空间，未初始化
x = t.Tensor(5, 3)
# print(x)
x = t.Tensor([[1,2], [3,4]])    # 此处是2维的张量
print(x)

# 0-1 之间随机数
x = t.rand(5, 3)
print(x)

# 查看x的形状
print(x.size())

# 查看x的形状 行和列 格式化输出
print('row:%d, col:%d' % (x.size()[0] ,x.size()[1]))

# 张量加法
c = t.add(t.rand(5, 3), t.rand(5, 3))
print(c)

d = t.rand(5, 3) + t.rand(5, 3)
print(d)

# 加法的特殊写法
result = t.Tensor(5, 3) # 预分配空间
t.add(c, d, out=result)
print(result)

# 张量的选取操作
print(result[:, 1])

# 张量与numpy之间的转换 注意：是共享内存的
a = t.ones(5)
print(type(a))

b = a.numpy()   # Tensor to Numpy
print(type(b))

e = np.ones(5)
f = t.from_numpy(e) # Numpy to Tensor
print(type(e), type(f))

# 取张量里某一个元素的值，先索引，然后item
scalar = f[0]
print(scalar.item())

# 张量交换数据 书中的说法有问题，并不能交换
t1 = t.Tensor([3,4])
old_t = t1
print(old_t)
new_t = t.Tensor(old_t)
new_t[0] = 111
print(old_t, new_t)

# tensor= > list
b = t.Tensor([[1,2,3],[4,5,6]])
# 转换为list类型
print(b.tolist())
# 判断张量大小
print(b.size())
# 获取张量元素数量
print(b.numel())

# 创建一个与b形状一样的张量
c = t.Tensor(b.size())
d = t.Tensor((2,3))

print(c.shape, d)

# 数字排列
print(t.arange(1, 6, 2))
print(t.linspace(1,10,3))
print(t.randperm(5))

# 单位矩阵  不要求行列数一样
print(t.eye(2,3))

# 改变形状
m = t.arange(0,6)
n = m.view(2,3)
print(n)

# 自动改变形状
n1 = m.view(-1,3)
print(n1)

n2 = n1.unsqueeze(-2)
print(n2)

# 索引操作
a =  t.randn(3,4)
print(a, a[0][2])
# 取前2行，前2列
print(a[:2, 0:2])

