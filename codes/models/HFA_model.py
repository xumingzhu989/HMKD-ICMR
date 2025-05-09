"""Pyramid Scene Parsing Network"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get_heterogeneous_feature_align_model']

from losses import CriterionKD
class RRB(nn.Module):

    def __init__(self, features, out_features):
        super(RRB, self).__init__()

        self.unify = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)
        self.residual = nn.Sequential(nn.Conv2d(out_features, out_features//4, kernel_size=3, padding=1, dilation=1, bias=False),
                                    nn.BatchNorm2d(out_features//4),
                                    nn.Conv2d(out_features//4, out_features, kernel_size=3, padding=1, dilation=1, bias=False))
        self.norm = nn.BatchNorm2d(out_features)

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        feats = self.norm(feats + residual)
        return feats
# ###
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        # 获取输入的尺寸
        batch_size, channels, height, width = x.size()
            
        # 全局平均池化
        squeeze = x.view(batch_size, channels, -1).mean(dim=2)  # (batch_size, channels)

        # 通过全连接层计算注意力权重
        excitation = self.fc1(squeeze)
        excitation = torch.relu(excitation)
        excitation = self.fc2(excitation)
        attention_weights = torch.sigmoid(excitation).view(batch_size, channels, 1, 1)

        # 缩放特征图
        return x * attention_weights
    
def compute_weighted_attention_map(feature_map):
    # 计算注意力图 (使用点积方法)
    batch_size, channels, height, width = feature_map.shape
    feature_map_flat = feature_map.view(batch_size, channels, -1)
    attention_scores = torch.bmm(feature_map_flat.permute(0, 2, 1), feature_map_flat)
    attention_map = F.softmax(attention_scores, dim=-1)

    # 计算加权特征图
    weighted_feature_map = torch.bmm(attention_map, feature_map_flat.permute(0, 2, 1))
    weighted_feature_map = weighted_feature_map.permute(0, 2, 1).view(batch_size, channels, height, width)

    return attention_map, weighted_feature_map

    # ###
class HFA_model(nn.Module):
    def __init__(self, batchsize):
        super(HFA_model, self).__init__()
        self.batchsize = batchsize
        self.conv_c=nn.Conv2d(320, 256, kernel_size=1)
        self.conv_c_2=nn.Conv2d(512, 512, kernel_size=1)#？
        self.conv_c_3=nn.Conv2d(320, 64, kernel_size=1)
        self.conv_c1010=nn.Conv2d(1024, 160, kernel_size=1)
        self.criterion = nn.MSELoss()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv1_04 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv2_04 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.delta_gen1=nn.Sequential(
                        nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False)
                        )
        self.delta_gen1_2 = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 2, kernel_size=3, padding=1, bias=False)
        )
        self.delta_gen1_04=nn.Sequential(
                        nn.Conv2d(160*2, 160, kernel_size=1, bias=False),
                        nn.BatchNorm2d(160),
                        nn.Conv2d(160, 2, kernel_size=3, padding=1, bias=False)
                        )
        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False)
        )
        self.delta_gen2_2 = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 2, kernel_size=3, padding=1, bias=False)
        )
        self.delta_gen2_04 = nn.Sequential(
            nn.Conv2d(160 * 2, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

        self.delta_gen1_2[2].weight.data.zero_()
        self.delta_gen2_2[2].weight.data.zero_()
        
        self.delta_gen1_04[2].weight.data.zero_()
        self.delta_gen2_04[2].weight.data.zero_()
        
        self.CA_fs = ChannelAttention(in_channels = 256)
        self.CA_ft = ChannelAttention(in_channels = 320)

        self.RRB_t = RRB(256, 256)
        self.RRB_s = RRB(256, 256)
        
        self.RRB_t_2 = RRB(512, 512)
        self.RRB_s_2 = RRB(512, 512)
        # self.RRB_s_2 = RRB(128, 512)
        
        self.RRB_t_04 = RRB(160, 160)
        self.RRB_s_04 = RRB(160, 160)
        
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w/s, h/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
    
    # def forward(self,f_t_3,f_s_3,f_t_4,f_s_4):
    # def forward(self,f_t_3,f_s_3,f_t1_3,f_s_1):
    def forward(self,f_t_3,f_s_3):
#         #f_t_3 B, 1024, 320 / B 32 32 C
#         #f_s_3 B, 256, 64, 64

#         # # upsample
#         # n, c, h, w = f_s_3.shape
#         # #print(n, c, h, w)
#         # f_t_3 = f_t_3.view(self.batchsize, int(h / 2), int(w / 2), -1).permute(0, 3, 1, 2)
#         # f_t_3 = F.interpolate(input=f_t_3, size=(h, w), mode='bilinear', align_corners=True)
#         # # 通过conv调整通道数一致
#         # f_t_3=self.conv_c(f_t_3)
#         # # #residual conv block
#         # # f_t_3=self.RRB_t(f_t_3)
#         # # f_s_3=self.RRB_s(f_s_3)
#         # #对位相乘
#         # h=torch.mul(f_s_3,f_t_3)
#         # #沿通道拼接
#         # h_t=self.conv1(h)
#         # h_s=self.conv2(h)
#         # h=torch.cat((h_t,h_s),dim=1)
#         # #生成offset
#         # d_t= self.delta_gen1(h)
#         # d_s= self.delta_gen2(h)
#         # #对齐异构特征
#         # f_t_3=self.bilinear_interpolate_torch_gridsample(f_t_3, (64, 64), d_t)
#         # f_s_3=self.bilinear_interpolate_torch_gridsample(f_s_3, (64, 64), d_s)
#         # #norm?
#         # loss=self.criterion(f_t_3,f_s_3)

#         # f_t_4 B, 256, 512 / B 16 16 C
#         # f_s_4 B, 512, 64, 64
#         # upsample

#         # n, c, h, w = f_s_4.shape
#         # # print(n, c, h, w)
#         # f_t_4 = f_t_4.view(self.batchsize, int(h / 4), int(w / 4), -1).permute(0, 3, 1, 2)
#         # f_t_4 = F.interpolate(input=f_t_4, size=(h, w), mode='bilinear', align_corners=True)
#         # # # 通过conv调整通道数一致
#         # # f_t_4 = self.conv_c_2(f_t_4)
#         # # # residual conv block
#         # # f_t_4 = self.RRB_t_2(f_t_4)
#         # # f_s_4 = self.RRB_s_2(f_s_4)
#         # # 对位相乘
#         # h = torch.mul(f_s_4, f_t_4)
#         # # 沿通道拼接
#         # h_t = self.conv1_2(h)
#         # h_s = self.conv2_2(h)
#         # h = torch.cat((h_t, h_s), dim=1)
#         # #print(h.shape)
#         # # 生成offset
#         # d_t = self.delta_gen1_2(h)
#         # d_s = self.delta_gen2_2(h)
#         # #print(d_t.shape,d_s.shape)
#         # # 对齐异构特征
#         # f_t_4 = self.bilinear_interpolate_torch_gridsample(f_t_4, (64, 64), d_t)
#         # f_s_4 = self.bilinear_interpolate_torch_gridsample(f_s_4, (64, 64), d_s)
#         # #norm?
#         # loss += self.criterion(f_t_4, f_s_4)
#         # #print(loss.item())
#         # ### 3
#         # upsample
#         n, c, h, w = f_s_3.shape
#             #print(n, c, h, w)
#         f_t_3 = f_t_3.view(self.batchsize, int(h / 2), int(w / 2), -1).permute(0, 3, 1, 2)
#             # ###学生模型特征图注意力
#         attention_map3, f_s_3 = compute_weighted_attention_map(f_s_3)
#         # 通过conv调整通道数一致
#         f_t_3 = F.interpolate(input=f_t_3, size=(h, w), mode='bilinear', align_corners=True)
#         # f_t_3 = self.deconv_layer_3(f_t_3)
#         f_t_3=self.conv_c(f_t_3)
#         f_t_3=self.RRB_t(f_t_3)
#         f_s_3=self.RRB_s(f_s_3)
#             #对位相乘
#         h=torch.mul(f_s_3,f_t_3)
#             #沿通道拼接
#         h_t=self.conv1(h)
#         h_s=self.conv2(h)
#         h=torch.cat((h_t,h_s),dim=1)
#             #生成offset
#         d_t= self.delta_gen1(h)
#         # d_s= self.delta_gen2(h)
#             #对齐异构特征
#         f_t_3=self.bilinear_interpolate_torch_gridsample(f_t_3, (64, 64), d_t)
#         # f_s_3=self.bilinear_interpolate_torch_gridsample(f_s_3, (64, 64), d_s)
#             #norm?
#         loss=self.criterion(f_t_3,f_s_3)
         
        # resnet101 b0
        # # torch.Size([4, 1024, 64, 64])
        # # torch.Size([4, 1024, 160])
        # # torch.Size([4, 1024, 45, 60])
        # # torch.Size([4, 690, 160])
        # # print(f_t_3.shape)
        # # print(f_s_3.shape)
        n, c, h, w = f_t_3.shape
        f_s_3 = f_s_3.view(self.batchsize, 23, 30, -1).permute(0, 3, 1, 2)
        # f_s_3 = f_s_3.view(self.batchsize, int(h / 2), int(w / 2), -1).permute(0, 3, 1, 2)
        attention_map3_t, f_t_3 = compute_weighted_attention_map(f_t_3)
        f_t_3 = F.interpolate(input=f_t_3, size=(23, 30), mode='bilinear', align_corners=True)
        # f_t_3 = F.interpolate(input=f_t_3, size=(int(h / 2), int(w / 2)), mode='bilinear', align_corners=True)
        f_t_3 = self.conv_c1010(f_t_3)
        f_t_3=self.RRB_t_04(f_t_3)
        f_s_3=self.RRB_s_04(f_s_3)
        h=torch.mul(f_s_3,f_t_3)
        # #沿通道拼接
        h_t=self.conv1_04(h)
        h_s=self.conv2_04(h)
        h=torch.cat((h_t,h_s),dim=1)
        #  #生成offset
        d_t= self.delta_gen1_04(h)
        # d_s= self.delta_gen2_04(h)
        f_t_3=self.bilinear_interpolate_torch_gridsample(f_t_3, (23, 30), d_t)
        # f_s_3=self.bilinear_interpolate_torch_gridsample(f_s_3, (32, 32), d_s)
        loss=self.criterion(f_t_3,f_s_3)
        
        # n1, c1, h1, w1 = f_s_1.shape
        # f_t1_3 = f_t1_3.view(self.batchsize, int(h1 / 4), int(w1 / 4), -1).permute(0, 3, 1, 2)
        # # 通过conv调整通道数一致
        # f_t1_3 = F.interpolate(input=f_t1_3, size=(h1, w1), mode='bilinear', align_corners=True)
        # f_t1_3=self.conv_c_3(f_t1_3)
        # loss2 = self.criterion(f_t1_3,f_s_1)
        
        # loss = loss1 + loss2
        return loss


def get_heterogeneous_feature_align_model(batchsize):
    model = HFA_model(batchsize)
    device = 'cuda'
    model.to(device)
    return model


if __name__ == '__main__':
    model = get_heterogeneous_feature_align_model(batchsize=4)
    f_t = torch.randn(4, 1024, 320).cuda()
    f_t_2 = torch.randn(4, 256, 512).cuda()
    f_s = torch.randn(4, 256, 64, 64).cuda()
    f_s_2 = torch.randn(4, 512, 64, 64).cuda()
    output = model(f_t, f_s,f_t_2,f_s_2)
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    print("Number of parameter: %.2fM" % (total / 1e6))

    for n,param in model.named_parameters():
        print(n,param.shape)

    # 输入fs,fc,gt返回cam和处理过的gt
