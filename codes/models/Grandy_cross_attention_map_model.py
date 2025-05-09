"""Pyramid Scene Parsing Network"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['get_Grandy_cross_attention_map_model']

from matplotlib import pyplot as plt


class Grandy_Cross_Attention_Map_model(nn.Module):
    def __init__(self,batchsize,h_t_3,w_t_3,h_s_3,w_s_3,h_t_4,w_t_4,h_s_4,w_s_4,h_gt,w_gt):
        super(Grandy_Cross_Attention_Map_model, self).__init__()

        self.batchsize=batchsize
        # self.h_t_1 = h_t_1
        # self.w_t_1 = w_t_1
        # self.h_s_1 = h_s_1
        # self.w_s_1 = w_s_1
        # self.h_t_2 = h_t_2
        # self.w_t_2 = w_t_2
        # self.h_s_2 = h_s_2
        # self.w_s_2 = w_s_2
        self.h_t_3 = h_t_3
        self.w_t_3 = w_t_3
        self.h_s_3 = h_s_3
        self.w_s_3 = w_s_3
        self.h_t_4 = h_t_4
        self.w_t_4 = w_t_4
        self.h_s_4 = h_s_4
        self.w_s_4 = w_s_4
        # self.h_t_cat = h_t_cat
        # self.w_t_cat = w_t_cat
        # self.h_s_aspp = h_s_aspp
        # self.w_s_aspp = w_s_aspp
        self.h_gt = h_gt
        self.w_gt = w_gt
        self.criterion_ATT = nn.MSELoss()


        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        # #stage1
        #
        # '''
        #         att_map_1_0.shape	torch.Size([2, 32768, 64])/128 256
        #         cnn_c1.shape	torch.Size([2, 64, 128, 256])
        #
        #         '''
        # self.upsample_1 = nn.Upsample(size=(h_t_1,w_t_1), mode='bilinear', align_corners=True)
        # self.ln_1= nn.LayerNorm(normalized_shape=[1, h_t_1, w_t_1])
        # self.lno_1 = nn.LayerNorm(normalized_shape=[1, h_t_1*w_t_1, h_t_1*w_t_1])
        # # self.fc_gt_1 = nn.Sequential(
        # #     nn.Linear(self.h_gt * self.w_gt, self.h_t_1 * self.w_t_1),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.conv_gt_1 = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=4, stride=4, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )
        # self.alpha_1=nn.Parameter(torch.tensor(1.0))
        # #stage2
        #
        # '''
        #         att_map_2_1.shape	torch.Size([2, 8192, 128])/64,128
        #         cnn_c2.shape	torch.Size([2, 128, 64, 128])
        #         '''
        # self.ln_2 = nn.LayerNorm(normalized_shape=[1, h_t_2, w_t_2])
        # self.lno_2 = nn.LayerNorm(normalized_shape=[1, h_t_2 * w_t_2, h_t_2 * w_t_2])
        # # self.fc_gt_2 = nn.Sequential(
        # #     nn.Linear(self.h_gt * self.w_gt, self.h_t_2 * self.w_t_2),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.conv_gt_2 = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=8, stride=8, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )
        # self.alpha_2 = nn.Parameter(torch.tensor(1.0))
        #stage3

        '''
        att_map_3_0.shape	torch.Size([2, 2048, 320])/32,64
        cnn_c3.shape	torch.Size([2, 256, 64, 128])
        '''
        self.conv_s_3=nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)
        self.ln_3 = nn.LayerNorm(normalized_shape=[1, h_t_3, w_t_3])
        self.lno_3 = nn.LayerNorm(normalized_shape=[1, h_t_3 * w_t_3, h_t_3 * w_t_3])
        # self.fc_gt_3 = nn.Sequential(
        #     nn.Linear(self.h_gt * self.w_gt, self.h_s_3 * self.w_s_3),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_gt_3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        #self.alpha_3 = nn.Parameter(torch.tensor(1.0))
        #stage4

        '''
                att_map_4_0.shape	torch.Size([2, 512, 512])/16,32
                cnn_c4.shape	torch.Size([2, 512, 64, 128])
        '''
        self.conv_s_4 = nn.Conv2d(1, 1, kernel_size=4, stride=4, padding=0)
        self.ln_4 = nn.LayerNorm(normalized_shape=[1, h_t_4, w_t_4])
        self.lno_4 = nn.LayerNorm(normalized_shape=[1, h_t_4 * w_t_4, h_t_4 * w_t_4])
        # self.fc_gt_4 = nn.Sequential(
        #     nn.Linear(self.h_gt * self.w_gt, self.h_s_4 * self.w_s_4),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_gt_4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=32, stride=32, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        #self.alpha_4 = nn.Parameter(torch.tensor(1.0))
        # #stage_final
        # '''
        #         _c.shape	torch.Size([2, 768, 128, 256])
        #         cnn_aspp.shape	torch.Size([2, 128, 64, 128])
        # '''
        # self.upsample_5 = nn.Upsample(size=(h_t_cat, w_t_cat), mode='bilinear', align_corners=True)
        # self.ln_5 = nn.LayerNorm(normalized_shape=[1, h_t_cat, w_t_cat])
        # self.lno_5 = nn.LayerNorm(normalized_shape=[1, h_t_cat * w_t_cat, h_t_cat * w_t_cat])
        # # self.fc_gt_5 = nn.Sequential(
        # #     nn.Linear(self.h_gt * self.w_gt, self.h_t_cat * self.w_t_cat),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.conv_gt_1 = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=4, stride=4, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )

        #self.conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)



    #def forward(self,attmap_1,attmap_2,attmap_3,attmap_4,_c,cnn_c1,cnn_c2,cnn_c3,cnn_c4,cnn_aspp,gt):
    def forward(self,  attmap_3, attmap_4,  cnn_c3, cnn_c4, gt):

        # #stage1
        #
        # '''
        # att_map_1_0.shape	torch.Size([2, 32768, 64])/128 256
        # cnn_c1.shape	torch.Size([2, 64, 128, 256])
        #
        # '''
        # attmap_1=attmap_1.view(self.batchsize, self.h_t_1, self.w_t_1, -1).permute(0, 3, 1, 2)
        # cnn_c1=cnn_c1.view(self.batchsize, self.h_s_1, self.w_s_1, -1).permute(0, 3, 1, 2)
        # #通道取均值
        # attmap_1=attmap_1.mean(dim=1).unsqueeze(1)#B,1,h_t_1,w_t_1
        # cnn_c1=cnn_c1.mean(dim=1).unsqueeze(1)#B,1,h_s_1,w_s_1
        # #f_c上采样
        # #cnn_c1_2=self.upsample_1(cnn_c1)#B,1,h_t_1,w_t_1
        # #层标准化
        # attmap_1 = self.ln_1(attmap_1)
        # cnn_c1 = self.ln_1(cnn_c1)
        # #hw*hw
        # attmap_1 = attmap_1.view(self.batchsize, 1, -1)
        # cnn_c1 = cnn_c1.view(self.batchsize, 1, -1)
        # o_1 = attmap_1.transpose(2, 1) * cnn_c1
        # o_1 = o_1.unsqueeze(1)#B,1,h_t_1*w_t_1,h_t_1*w_t_1
        # #output通过conv2
        # o_1=self.conv2(o_1)#B,1,h_t_1*w_t_1,h_t_1*w_t_1
        # o_1=self.lno_1(o_1)
        #
        #
        # # B,1,512,1024
        # gt_1=self.conv_gt_1(gt)#B,1,h_t_1,w_t_1
        # gt_1=self.ln_1(gt_1)
        # gt_1=gt_1.view(self.batchsize, 1, -1)
        # gt_1=gt_1.transpose(1, 2)*gt_1
        # gt_1=gt_1.unsqueeze(1)#B,1,h_t_1*w_t_1,h_t_1*w_t_1
        # gt_1=self.lno_1(gt_1)
        # loss_1=self.alpha_1*self.criterion_ATT(o_1,gt_1)
        #
        #
        # #stage2
        # '''
        # att_map_2_1.shape	torch.Size([2, 8192, 128])/64,128
        # cnn_c2.shape	torch.Size([2, 128, 64, 128])
        # '''
        #
        # attmap_2 = attmap_2.view(self.batchsize, self.h_t_2, self.w_t_2, -1).permute(0, 3, 1, 2)
        # cnn_c2 = cnn_c2.view(self.batchsize, self.h_s_2, self.w_s_2, -1).permute(0, 3, 1, 2)
        # # 通道取均值
        # attmap_2 = attmap_2.mean(dim=1).unsqueeze(1)  # B,1,h_t_2,w_t_2
        # cnn_c2 = cnn_c2.mean(dim=1).unsqueeze(1)  # B,1,h_s_2,w_s_2
        # # 层标准化
        # attmap_2 = self.ln_2(attmap_2)
        # cnn_c2 = self.ln_2(cnn_c2)
        # # hw*hw
        # attmap_2 = attmap_2.view(self.batchsize, 1, -1)
        # cnn_c2 = cnn_c2.view(self.batchsize, 1, -1)
        # o_2 = attmap_2.transpose(2, 1) * cnn_c2
        # o_2 = o_2.unsqueeze(1)  # B,1,h_t_2*w_t_2,h_t_2*w_t_2
        # # output通过conv2
        # o_2 = self.conv2(o_2)  # B,1,h_t_2*w_t_2,h_t_2*w_t_2
        # o_2 = self.lno_2(o_2)
        #
        # # gt通过全连接层和大小一致
        # # gt_f = gt.view(gt.size(0), -1)
        # # gt_2 = self.fc_gt_2(gt_f)
        # # gt_2 = gt_2.unsqueeze(1)
        # # my_gt_2 = gt_2.transpose(1, 2) * gt_2
        # # my_gt_2 = my_gt_2.unsqueeze(1)  # B,1,h_t_2*w_t_2,h_t_2*w_t_2
        # # my_gt_2 = self.lno_2(my_gt_2)
        # gt_2 = self.conv_gt_2(gt)  # B,1,h_t_2,w_t_2
        # gt_2 = self.ln_2(gt_2)
        # gt_2 = gt_2.view(self.batchsize, 1, -1)
        # gt_2 = gt_2.transpose(1, 2) * gt_2
        # gt_2 = gt_2.unsqueeze(1)  # B,1,h_t_2*w_t_2,h_t_2*w_t_2
        # gt_2 = self.lno_2(gt_2)
        # loss_2 = self.alpha_2 * self.criterion_ATT(o_2, gt_2)


        #stage3


        attmap_3 = attmap_3.view(self.batchsize, self.h_t_3, self.w_t_3, -1).permute(0, 3, 1, 2)
        cnn_c3 = cnn_c3.view(self.batchsize, self.h_s_3, self.w_s_3, -1).permute(0, 3, 1, 2)
        # 通道取均值
        attmap_3 = attmap_3.mean(dim=1).unsqueeze(1)  # B,1,h_t_3,w_t_3
        cnn_c3 = cnn_c3.mean(dim=1).unsqueeze(1)  # B,1,h_s_3,w_s_3
        # f_c上采样
        cnn_c3 = self.conv_s_3(cnn_c3)  # B,1,h_t_3,w_t_3
        # 层标准化
        attmap_3 = self.ln_3(attmap_3)
        cnn_c3 = self.ln_3(cnn_c3)
        # hw*hw
        attmap_3 = attmap_3.view(self.batchsize, 1, -1)
        cnn_c3 = cnn_c3.view(self.batchsize, 1, -1)
        o_3 = attmap_3.transpose(2, 1) * cnn_c3
        o_3 = o_3.unsqueeze(1)  # B,1,h_t_3*w_t_3,h_t_3*w_t_3
        # output通过conv2
        o_3 = self.conv2(o_3)  # B,1,h_t_3*w_t_3,h_t_3*w_t_3
        o_3 = self.lno_3(o_3)

        # gt通过全连接层和大小一致
        # gt_f = gt.view(gt.size(0), -1)
        # gt_3 = self.fc_gt_3(gt_f)
        # gt_3 = gt_3.unsqueeze(1)
        # my_gt_3 = gt_3.transpose(1, 2) * gt_3
        # my_gt_3 = my_gt_3.unsqueeze(1)  # B,1,h_s_3*w_s_3,h_s_3*w_s_3
        # my_gt_3 = self.lno_3(my_gt_3)
        gt_3 = self.conv_gt_3(gt)  # B,1,h_s_3,w_s_3
        gt_3 = self.ln_3(gt_3)
        gt_3 = gt_3.view(self.batchsize, 1, -1)
        gt_3 = gt_3.transpose(1, 2) * gt_3
        gt_3 = gt_3.unsqueeze(1)  # B,1,h_s_3*w_s_3,h_s_3*w_s_3
        gt_3 = self.lno_3(gt_3)
        loss_3 = self.criterion_ATT(o_3, gt_3)


        #stage4

        attmap_4 = attmap_4.view(self.batchsize, self.h_t_4, self.w_t_4, -1).permute(0, 3, 1, 2)
        cnn_c4 = cnn_c4.view(self.batchsize, self.h_s_4, self.w_s_4, -1).permute(0, 3, 1, 2)
        # 通道取均值
        attmap_4 = attmap_4.mean(dim=1).unsqueeze(1)  # B,1,h_t_4,w_t_4
        cnn_c4 = cnn_c4.mean(dim=1).unsqueeze(1)  # B,1,h_s_4,w_s_4
        # 上采样
        cnn_c4 = self.conv_s_4(cnn_c4)  # B,1,h_t_4,w_t_4
        # 层标准化
        attmap_4 = self.ln_4(attmap_4)
        cnn_c4 = self.ln_4(cnn_c4)
        # hw*hw
        attmap_4 = attmap_4.view(self.batchsize, 1, -1)
        cnn_c4 = cnn_c4.view(self.batchsize, 1, -1)
        o_4 = attmap_4.transpose(2, 1) * cnn_c4
        o_4 = o_4.unsqueeze(1)  # B,1,h_s_4*w_s_4,h_s_4*w_s_4
        # output通过conv2
        o_4 = self.conv2(o_4)  # B,1,h_s_4*w_s_4,h_s_4*w_s_4
        o_4 = self.lno_4(o_4)

        # gt通过全连接层和大小一致
        # gt_f = gt.view(gt.size(0), -1)
        # gt_4 = self.fc_gt_4(gt_f)
        # gt_4 = gt_4.unsqueeze(1)
        # my_gt_4 = gt_4.transpose(1, 2) * gt_4
        # my_gt_4 = my_gt_4.unsqueeze(1)  #B,1,h_s_4*w_s_4,h_s_4*w_s_4
        # my_gt_4 = self.lno_4(my_gt_4)
        gt_4 = self.conv_gt_4(gt)  # B,1,h_s_4,w_s_4
        gt_4 = self.ln_4(gt_4)
        gt_4 = gt_4.view(self.batchsize, 1, -1)
        gt_4 = gt_4.transpose(1, 2) * gt_4
        gt_4 = gt_4.unsqueeze(1)  # B,1,h_s_4*w_s_4,h_s_4*w_s_4
        gt_4 = self.lno_4(gt_4)
        loss_4 = self.criterion_ATT(o_4, gt_4)

        #
        # #aspp
        # print('aspp')
        # '''
        # _c.shape	torch.Size([2, 768, 128, 256])
        # cnn_aspp.shape	torch.Size([2, 128, 64, 128])
        # '''
        # _c = _c.view(self.batchsize, self.h_t_cat, self.w_t_cat, -1).permute(0, 3, 1, 2)
        # cnn_aspp = cnn_aspp.view(self.batchsize, self.h_s_aspp, self.w_s_aspp, -1).permute(0, 3, 1, 2)
        # # 通道取均值
        # _c = _c.mean(dim=1).unsqueeze(1)  # B,1,h_t_cat,w_t_cat
        # cnn_aspp = cnn_aspp.mean(dim=1).unsqueeze(1)  # B,1,h_s_aspp,w_s_aspp
        # # 上采样
        # cnn_aspp = self.upsample_5(cnn_aspp)  # B,1,h_t_cat,w_t_cat
        # # 层标准化
        # f_t_ln = self.ln_5(_c)
        # f_s_ln = self.ln_5(cnn_aspp)
        # # hw*hw
        # f_t_r = f_t_ln.view(self.batchsize, 1, -1)
        # f_s_r = f_s_ln.view(self.batchsize, 1, -1)
        # o_5 = f_t_r.transpose(2, 1) * f_s_r
        # o_5 = o_5.unsqueeze(1)  # B,1,h_t_cat*w_t_cat,h_t_cat*w_t_cat
        # # output通过conv2
        # o_5 = self.conv2(o_5)  # B,1,h_t_cat*w_t_cat,h_t_cat*w_t_cat
        # o_5 = self.lno_5(o_5)
        #
        # # gt通过全连接层和大小一致
        # # gt_f = gt.view(gt.size(0), -1)
        # # gt_5 = self.fc_gt_5(gt_f)
        # # gt_5 = gt_5.unsqueeze(1)
        # # my_gt_5 = gt_5.transpose(1, 2) * gt_5
        # # my_gt_5 = my_gt_5.unsqueeze(1)  # B,1,h_t_cat*w_t_cat,h_t_cat*w_t_cat
        # # my_gt_5 = self.lno_5(my_gt_5)
        # gt_5 = self.conv_gt_5(gt)  # B,1,h_t_cat,w_t_cat
        # gt_5 = self.ln_5(gt_5)
        # gt_5 = gt_5.view(self.batchsize, 1, -1)
        # gt_5 = gt_5.transpose(1, 2) * gt_5
        # gt_5 = gt_5.unsqueeze(1)  # B,1,h_t_cat*w_t_cat,h_t_cat*w_t_cat
        # gt_5 = self.lno_4(gt_5)
        # print(o_5.shape)
        # print(gt_5.shape)




        return [loss_3,loss_4]



def get_Grandy_cross_attention_map_model(batchsize,h_t_3,w_t_3,h_s_3,w_s_3,h_t_4,w_t_4,h_s_4,w_s_4,h_gt,w_gt):
    model = Grandy_Cross_Attention_Map_model(batchsize,h_t_3,w_t_3,h_s_3,w_s_3,h_t_4,w_t_4,h_s_4,w_s_4,h_gt,w_gt)

    return model


if __name__ == '__main__':
    # a=torch.tensor(1)
    # print(sum([a,a,a]))
    # sys.exit()
    # attmap_1 = torch.randn(2, 32768, 64)
    # attmap_2 = torch.randn(2, 8192, 128)
    attmap_3 = torch.randn(2, 2048, 320)
    attmap_4 = torch.randn(2, 512, 512)
    #_c = torch.randn(2, 768, 128, 256)
    # cnn_c1 = torch.randn(2, 64, 128, 256)
    # cnn_c2 = torch.randn(2, 128, 64, 128)
    cnn_c3 = torch.randn(2, 256, 64, 128)
    cnn_c4 = torch.randn(2, 512, 64, 128)
    #cnn_aspp = torch.randn(2, 128, 64, 128)
    gt = torch.ones(2, 1, 512, 1024)
    model = get_Grandy_cross_attention_map_model(batchsize=2,h_t_3=32,w_t_3=64,h_s_3=64,w_s_3=128,h_t_4=16,w_t_4=32,h_s_4=64,w_s_4=128,h_gt=512,w_gt=1024)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {num_params}")
    #print(model)
    output=model(attmap_3,attmap_4,cnn_c3,cnn_c4,gt)
    print(output)

