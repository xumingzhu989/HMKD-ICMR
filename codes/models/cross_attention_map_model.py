"""Pyramid Scene Parsing Network"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['get_cross_attention_map_model']


class Cross_Attention_Map_model(nn.Module):
    def __init__(self,batchsize,h_s,w_s,h_c,w_c,h_gt,w_gt):
        super(Cross_Attention_Map_model, self).__init__()
        self.batchsize=batchsize
        self.h_s = h_s
        self.w_s = w_s
        self.h_c=h_c
        self.w_c=w_c
        self.h_gt = h_gt
        self.w_gt = w_gt
        self.upsample = nn.Upsample(size=(h_s, w_s), mode='bilinear', align_corners=True)
        self.ln= nn.LayerNorm(normalized_shape=[1, h_s, w_s])
        self.ln_2 = nn.LayerNorm(normalized_shape=[1, h_s*w_s, h_s*w_s])
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.fc_gt = nn.Sequential(
            nn.Linear(self.h_gt * self.w_gt, self.h_s * self.w_s),
            nn.ReLU(inplace=True)
        )
        #self.conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)



    def forward(self,fs,fc,gt):

        fs=fs.view(self.batchsize, self.h_s, self.w_s, -1).permute(0, 3, 1, 2)
        #通道取均值
        fs_1=fs.mean(dim=1).unsqueeze(1)#B,1,h_s,w_s
        fc_1=fc.mean(dim=1).unsqueeze(1)#B,1,h_c,w_c
        #f_c上采样
        fc_1=self.upsample(fc_1)#B,1,h_s,w_s
        #层标准化
        fs_ln = self.ln(fs_1)
        fc_ln = self.ln(fc_1)
        #hw*hw
        fs_r = fs_1.view(self.batchsize, 1, -1)
        fc_r = fc_1.view(self.batchsize, 1, -1)
        output = fs_r.transpose(2, 1) * fc_r
        output = output.unsqueeze(1)#B,1,h_s*w_s,h_s*w_s
        #output通过conv2
        output=self.conv2(output)#B,1,h_s*w_s,h_s*w_s
        output=self.ln_2(output)




        #gt通过全连接层和fc_r大小一致
        gt_f=gt.view(gt.size(0), -1)

        gt_1=self.fc_gt(gt_f)

        gt_2=gt_1.unsqueeze(1)

        my_gt=gt_2.transpose(1, 2)*gt_2

        my_gt=my_gt.unsqueeze(1)#B,1,h_s*w_s,h_s*w_s
        my_gt=self.ln_2(my_gt)


        return output,my_gt




def get_cross_attention_map_model(batchsize,h_s,w_s,h_c,w_c,h_gt,w_gt):
    model = Cross_Attention_Map_model(batchsize,h_s,w_s,h_c,w_c,h_gt,w_gt)
    device='cuda'
    model.to(device)
    return model


if __name__ == '__main__':
    model = get_cross_attention_map_model(batchsize=2,h_s=16,w_s=32,h_c=64,w_c=128,h_gt=512,w_gt=1024)
    gt=torch.ones(2, 1,512, 1024).cuda()
    fs=torch.randn(2, 512, 512).cuda()
    fc=torch.randn(2, 128, 64, 128).cuda()
    #gt=torch.ones(8, 1, 512, 1024).cuda()#B,512,1024
    output = model(fs,fc,gt)
    #输入fs,fc,gt返回cam和处理过的gt
