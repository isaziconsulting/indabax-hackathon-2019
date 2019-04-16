import torch
import torchvision

x = torch.rand(5, 3)
print(x)
print("If you can see this, your PyTorch installation is working!")

try:
    torch.cuda.is_available()
    print("CUDA is working!")
except:
    print("ERROR: could not access GPU or CUDA")