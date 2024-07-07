import torch
from torchvision import models

"""
    More detail about the VGG architecture (if you want to understand magic/hardcoded numbers) can be found here:

    https://github.com/pytorch/vision/blob/3c254fb7af5f8af252c24e89949c54a3461ff0be/torchvision/models/vgg.py
"""


class Vgg16(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(
            pretrained=True, progress=show_progress
        ).features
        self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        self.content_feat_idx = 1  # relu2_2
        self.style_feat_idxs = list(
            range(len(self.layer_names))
        )  # all layers used for style representation

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x

        return (relu1_2, relu2_2, relu3_3, relu4_3)


class Vgg19(torch.nn.Module):
    """
    Used in the original NST paper, only those layers are exposed which were used in the original paper

    'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' were used for style representation
    'conv4_2' was used for content representation (although they did some experiments with conv2_2 and conv5_2)
    """

    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        vgg_pretrained_features = models.vgg19(
            pretrained=True, progress=show_progress
        ).features
        if use_relu:  # use relu or as in original paper conv layers
            self.layer_names = [
                "relu1_1",
                "relu2_1",
                "relu3_1",
                "relu4_1",
                "conv4_2",
                "relu5_1",
            ]
            self.offset = 1
        else:
            self.layer_names = [
                "conv1_1",
                "conv2_1",
                "conv3_1",
                "conv4_1",
                "conv4_2",
                "conv5_1",
            ]
            self.offset = 0
        self.content_feat_idx = 4  # conv4_2
        # all layers used for style representation except conv4_2
        self.style_feat_idxs = list(range(len(self.layer_names)))
        self.style_feat_idxs.remove(4)  # conv4_2

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1 + self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1 + self.offset, 6 + self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6 + self.offset, 11 + self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11 + self.offset, 20 + self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20 + self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29 + +self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x

        return (layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
