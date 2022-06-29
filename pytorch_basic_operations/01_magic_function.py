# dir()函数：打开，看见(让我们知道工具箱以及工具箱的分隔区有什么东西)
# help()函数：说明书(让我们知道每个工具是如何使用的，即工具箱的使用方法)

import torch

print(dir(torch))
print(dir(torch.cuda))
print(dir(torch.cuda.is_available()))
print(help(torch.cuda.is_available))
