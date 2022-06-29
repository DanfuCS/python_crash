# 模型的保存和模型的加载
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1: 保存模型的结构和参数
torch.save(vgg16, "vgg16_method1.pth")

# 加载方式1
model = torch.load("vgg16_method1.pth")
print(model)

# 保存方式2: 仅保存模型参数(推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 加载方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

