import torch as t
import numpy as np

# 为tensor设置requires_grad标识，代表着需要求导数
# pytorch会自动调用autograd记录操作
x = t.ones(2, 2, requires_grad= True)

print(x)

y = x.sum()
print(y)

print(y.grad_fn)

y.backward() # 反向传播，计算梯度

print(x.grad)

y.backward() # 反向传播，再次计算梯度

print(x.grad)

# 注意：grad反向传播过程是累加的，如果不需要累加可以自动清零
x.grad.data.zero_()

y.backward() # 反向传播，再次计算梯度

print(x.grad)