"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

__all__ = ['get_deeplabv3']


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
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

    def __init__(self, nclass, backbone='resnet50', aux=False, local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        # print('student begin:sf1,sf2,sf3,sf4')
        # print('cnn_c1.shape\t{}'.format(c1.shape))
        # print('cnn_c2.shape\t{}'.format(c2.shape))
        # print('cnn_c3.shape\t{}'.format(c3.shape))
        # print('cnn_c4.shape\t{}'.format(c4.shape))
        # print('student over:sf1,sf2,sf3,sf4')
        '''
        [512,512]
        student begin:sf1,sf2,sf3,sf4
        cnn_c1.shape    torch.Size([4, 64, 128, 128])
        cnn_c2.shape    torch.Size([4, 128, 64, 64])
        cnn_c3.shape    torch.Size([4, 256, 64, 64])
        cnn_c4.shape    torch.Size([4, 512, 64, 64])
        student over:sf1,sf2,sf3,sf4
        student begin:aspp
        cnn_aspp.shape  torch.Size([4, 128, 64, 64])
        student over:aspp
        '''
        '''
        [512,1024]
        student begin:sf1,sf2,sf3,sf4
        cnn_c1.shape    torch.Size([4, 64, 128, 256])
        cnn_c2.shape    torch.Size([4, 128, 64, 128])
        cnn_c3.shape    torch.Size([4, 256, 64, 128])
        cnn_c4.shape    torch.Size([4, 512, 64, 128])
        student over:sf1,sf2,sf3,sf4
        student begin:aspp
        cnn_aspp.shape  torch.Size([4, 128, 64, 128])
        student over:aspp
        '''
        # print('cnn_c1.shape\t{}'.format(c1.shape))
        # torch.save(c1, 'cnn_c1.pt')
        # print('cnn_c2.shape\t{}'.format(c2.shape))
        # torch.save(c2, 'cnn_c2.pt')
        # print('cnn_c3.shape\t{}'.format(c3.shape))
        # torch.save(c3, 'cnn_c3.pt')
        # print('cnn_c4.shape\t{}'.format(c4.shape))
        # torch.save(c4, 'cnn_c4.pt')

        x, x_feat_after_aspp = self.head(c4)
        '''
       
        '''
        # print('student begin:aspp')
        # print('cnn_aspp.shape\t{}'.format(x_feat_after_aspp.shape))
        # print('student over:aspp')
        
        # torch.save(x_feat_after_aspp, 'cnn_aspp.pt')

        # if self.aux:
        #   auxout = self.auxlayer(c3)
        #return [x, auxout, x_feat_after_aspp]
        # 返回通道平均之后的注意力图

        # c1 = torch.mean(c1, dim=1).unsqueeze(1)
        # c2 = torch.mean(c2, dim=1).unsqueeze(1)
        # c3 = torch.mean(c3, dim=1).unsqueeze(1)
        # c4 = torch.mean(c4, dim=1).unsqueeze(1)


        return [x,c1,c2,c3,c4,x_feat_after_aspp]



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
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
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
            nn.Conv2d(out_channels, nclass, 1)
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


def get_deeplabv3(backbone='resnet50', local_rank=None, pretrained=None,
                  pretrained_base=True, num_class=19, **kwargs):
    model = DeepLabV3(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        print(local_rank)
        if local_rank is not None:
            print('enter local rank')
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
        #model.load_state_dict(torch.load(pretrained))
    # print('get_deeplabv3')    
    # print(dir(backbone)) # 应该是 <class 'models.deeplabv3.DeepLabV3'>
    # print('get_deeplabv3')    
    return model


if __name__ == '__main__':
    model = get_deeplabv3()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)