"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#from .segbase import SegBaseModel

__all__ = ['get_deeplabv3']

from CIRKD1127.models.segbase_my import SegBaseModel


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    num_class : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, num_class, backbone='resnet50', local_rank=None, pretrained=True,pretrained_base='', **kwargs):
        super(DeepLabV3, self).__init__(num_class, backbone, local_rank, pretrained,pretrained_base, **kwargs)
        self.backbone=backbone
        if backbone == 'resnet18':

            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, num_class, **kwargs)


        #self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
        self.__setattr__('exclusive',['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        x, x_feat_after_aspp = self.head(c4)

        # if self.aux:
        #     auxout = self.auxlayer(c3)
        return [x, x_feat_after_aspp]


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_class, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(in_channels, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels,num_class, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.block[0:4](x)
        x_feat_after_aspp = x
        x = self.block[4](x)
        return x, x_feat_after_aspp


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_deeplabv3(num_class, backbone, local_rank, pretrained, pretrained_base, **kwargs):
    # import pdb
    # pdb.set_trace()
    model = DeepLabV3(num_class=num_class, backbone=backbone, local_rank=local_rank, pretrained=pretrained, pretrained_base=pretrained_base, **kwargs)

    if pretrained != 'None':
        print('pretrained')
        if local_rank is not None:
            model.load_state_dict(torch.load(pretrained_base, map_location=torch.device(local_rank)))
        else:
            model.load_state_dict(torch.load(pretrained_base))

    return model


# def get_deeplabv3(num_class, backbone, local_rank, pretrained, pretrained_base, **kwargs):
#     model = DeepLabV3(num_class=num_class, backbone=backbone, local_rank=local_rank, pretrained=pretrained,
#                       pretrained_base=pretrained_base, **kwargs)
#     print(type(model))
#     if pretrained != 'None':
#         if local_rank is not None:
#             pretrained_dict = torch.load(pretrained_base, map_location=torch.device(local_rank))
#         else:
#             pretrained_dict = torch.load(pretrained_base)
#
#         model_dict = model.state_dict()
#
#         # Create a new dictionary that contains only matching keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#
#         # Load the pretrained weights to the matching keys in the model
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#
#         # Print out the loaded keys
#         print("Loaded keys:")
#         for k in pretrained_dict.keys():
#             print(k)
#
#
#     return model


if __name__ == '__main__':
    model = get_deeplabv3(num_class=19,backbone='resnet18', local_rank=None, pretrained='None',pretrained_base='resnet18-imagenet.pth')

    img = torch.randn(2, 3, 480, 480)
    output = model(img)