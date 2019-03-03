import torch
from torch import nn
from model.build_contextpath import build_contextpath


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(self.conv1(input)))  # CBR


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=16)  # less
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32)
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64)
        # self.convblock1 = ConvBlock(in_channels=4, out_channels=64)  # todo: 3->4
        # self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        # self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    # 3 layers
    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x  # 1/8, [1, 256, 60, 80]


# ARM
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, phase_train=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # in = out
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.phase_train = phase_train

    def forward(self, input):
        # global average pooling
        x = torch.mean(input, dim=3, keepdim=True)  # first cols get mean, then rows get mean
        x = torch.mean(x, dim=2, keepdim=True)  # fm -> one value, fms -> vector
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        if self.phase_train:  # when test, no need BN
            x = self.bn(x)
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


# FFM
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 18_s
        self.in_channels = 448  # 64 + (128 + 256)
        # 18/34
        # self.in_channels = 832  # 64 + (256 + 512)
        # 50
        # self.in_channels = 3328  # 256 + (1024 + 2048)
        # ConvBlock 3x3 kernel
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)  # 3328->32
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        # global average pool
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class MidSupervisionModule(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.convblock = ConvBlock(in_channels=in_channels, out_channels=num_classes, stride=1)  # 1024->38, 2048->38
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feature = self.convblock(input)  # 1024 -> 38 fm  torchSize([1, 38, 30, 40])
        # global average pool, a fm -> a value
        x = torch.mean(feature, 3, keepdim=True)  # torch.Size([1, 38, 30, 1])
        x = torch.mean(x, 2, keepdim=True)  # torch.Size([1, 38, 1, 1])
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)  # feature: cbr output
        x = torch.add(x, feature)  # [1, 38, 30, 40]
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path, phase_train=False):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path, pretrained=True)

        # build attention refinement module
        # 50_s
        self.attention_refinement_module1 = AttentionRefinementModule(128, 128, phase_train=phase_train)  # in = out
        self.attention_refinement_module2 = AttentionRefinementModule(256, 256, phase_train=phase_train)
        # 34/18
        # self.attention_refinement_module1 = AttentionRefinementModule(256, 256, phase_train=phase_train)  # in = out
        # self.attention_refinement_module2 = AttentionRefinementModule(512, 512, phase_train=phase_train)
        # 50
        # self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024, phase_train=phase_train)  # in = out
        # self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048, phase_train=phase_train)

        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes)

        # train, add mid supervision
        self.phase_train = phase_train
        # 50_s
        self.cx1_mid_supervision_module = MidSupervisionModule(in_channels=128, num_classes=num_classes)
        self.cx2_mid_supervision_module = MidSupervisionModule(in_channels=256, num_classes=num_classes)
        # 34/18
        # self.cx1_mid_supervision_module = MidSupervisionModule(in_channels=256, num_classes=num_classes)
        # self.cx2_mid_supervision_module = MidSupervisionModule(in_channels=512, num_classes=num_classes)
        # 50
        # self.cx1_mid_supervision_module = MidSupervisionModule(in_channels=1024, num_classes=num_classes)
        # self.cx2_mid_supervision_module = MidSupervisionModule(in_channels=2048, num_classes=num_classes)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input[:, :3, :, :])
        # sx = self.saptial_path(input)  # 4 channel

        # output of context path
        cx1, cx2, tail = self.context_path(input)  # tail 1/32 fm global average
        cx1 = self.attention_refinement_module1(cx1)  # 1/16
        cx2 = self.attention_refinement_module2(cx2)  # 1/32
        cx2 = torch.mul(cx2, tail)

        if self.phase_train:  # should be here, because code below will change cx1,cx2
            result_16 = self.cx1_mid_supervision_module(cx1)  # 1/16 [1, 38, 30, 40]
            result_32 = self.cx2_mid_supervision_module(cx2)  # 1/32 [1, 38, 15, 20]

        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)  # context path output (2048+1024) 1/8, 1/8

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)  # 1/8 [1, 38, 60, 80]
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')  # 1,1
        result = self.conv(result)  # [1, 38, 480, 640]

        if self.phase_train:
            return result, result_16, result_32
        else:
            return result
