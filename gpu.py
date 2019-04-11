import torch as t

tensor = t.Tensor(3, 4)

print(tensor.cuda(0))
print(tensor.is_cuda)

tensor = tensor.cuda()
print(tensor.is_cuda)

