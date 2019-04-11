import torch

print(torch.__version__)
print('gpu:', torch.cuda.is_available())        # 没有安装gpu版本的torch