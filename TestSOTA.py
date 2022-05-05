from model.VGG16 import vgg
import torch

if __name__ == '__main__':
    VGG = vgg("vgg16", 10)
    x = torch.rand(size=(8, 2, 64, 64))
    output = VGG(x)
    print(output.size())