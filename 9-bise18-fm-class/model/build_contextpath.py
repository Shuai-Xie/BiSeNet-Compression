import torch
from model import resnet_s


def build_contextpath(name, pretrained=True):
    model = {
        'resnet18_s': resnet_s.resnet18_s(),
        'resnet50_s': resnet_s.resnet50_s()
    }
    return model[name]


if __name__ == '__main__':
    model = build_contextpath('resnet18_s', pretrained=False)
    # img = torch.rand(1, 4, 480, 640)
    img = torch.rand(1, 4, 256, 320)
    f3, f4, tail = model.forward(img)
    print(f3.size())  # [1, 64, 15, 20]
    print(f4.size())  # [1, 128, 8, 10]
    print(tail.size())
