import torch as t

a = t.arange(0, 6).view(2, 3)
print(a, t.cos(a))
print(a%3)
print(a**2)
print(t.clamp(a, min=3, max=4))

b=t.ones(2,3)
print(b.sum(dim=0, keepdim=False))
