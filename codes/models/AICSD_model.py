"""Pyramid Scene Parsing Network"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
import math
from scipy.stats import norm

__all__ = ['get_AICSD_model']
def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8
def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])
def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis
def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)
def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)
class AICSD_model(nn.Module):
    def __init__(self, batchsize):
        super(AICSD_model, self).__init__()
        # t_channels = t_net.get_channel_num()
        # s_channels = s_net.get_channel_num()

        # self.t_net = t_net
        # self.s_net = s_net
        #self.loss_divider = [8, 4, 2, 1, 1, 4 * 4]
        self.batchsize = batchsize
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.lo_lambda = 1
        self.pi_lambda = 1
        self.conv_c=nn.Conv2d(128, 768, kernel_size=1)
        # self.conv_c1=nn.Conv2d(768, 128, kernel_size=1)
        
        #self.scale = 0.5

    #def forward(self, fs, ft):  # 修改
    def forward(self, s_out, t_out, t_logit, s_logit):  # 修改s_out,t_out是
        pi_loss = 0 # 10 YES
        #if self.args.pi_lambda is not None:  # pixelwise loss
        if self.pi_lambda is not None:
            B ,C ,H, W = t_out.size()
            s_out1 = F.interpolate(s_out, (H, W), mode='bilinear', align_corners=True)
            s_out1 = self.conv_c(s_out1)
            # print(s_out.shape)
            # print(t_out.shape)
            # TF = F.normalize(t_feats[5].pow(2).mean(1))
            # SF = F.normalize(s_feats[5].pow(2).mean(1))
            # pi_loss = self.args.pi_lambda * (TF - SF).pow(2).mean()
            pi_loss = self.pi_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out1 / self.temperature, dim=1),
                                                                 F.softmax(t_out / self.temperature, dim=1))
        
        lo_loss = 0  # 对应的损失函数主要，s_out以及t_out为最后一层输出的特征图，在softmax之前的那一个 YES
        if self.lo_lambda is not None:
            b, n, h, w = s_logit.size()
            # print(s_out.shape)
            t_logit = F.interpolate(t_logit, (h, w), mode='bilinear', align_corners=True)
            # t_logit = self.conv_c1(t_logit)
            # print(t_out1.shape)
            s_logit = torch.reshape(s_logit, (b, n, h * w))
            t_logit = torch.reshape(t_logit, (b, n, h * w))

            s_logit = F.softmax(s_logit / self.temperature, dim=2)
            t_logit = F.softmax(t_logit / self.temperature, dim=2)
            # print(s_logit.shape)
            kl = torch.nn.KLDivLoss(reduction="batchmean")
            ICCS = torch.empty((19, 19)).cuda()
            ICCT = torch.empty((19, 19)).cuda()
            for i in range(19):
                for j in range(i, 19):
                    ICCS[j, i] = ICCS[i, j] = kl(s_logit[:, i], s_logit[:, j])
                    ICCT[j, i] = ICCT[i, j] = kl(t_logit[:, i], t_logit[:, j])

            ICCS = torch.nn.functional.normalize(ICCS, dim=1)
            ICCT = torch.nn.functional.normalize(ICCT, dim=1)
            # lo_loss = self.lo_lambda * (ICCS - ICCT).pow(2).mean() / b
            loss1 = (ICCS - ICCT).pow(2)
            lo_loss = self.lo_lambda * loss1.sum()
        #if self.args.lo_lambda is not None:
#         if self.lo_lambda is not None:
#             b, c, h, w = s_out.size()
#             # print(s_out.shape)
#             t_out1 = F.interpolate(t_out, (h, w), mode='bilinear', align_corners=True)
#             t_out1 = self.conv_c1(t_out1)
#             # print(t_out1.shape)
#             s_logit = torch.reshape(s_out, (b, c, h * w))
#             t_logit = torch.reshape(t_out1, (b, c, h * w))

#             s_logit = F.softmax(s_logit / self.temperature, dim=2)
#             t_logit = F.softmax(t_logit / self.temperature, dim=2)
#             # print(s_logit.shape)
#             kl = torch.nn.KLDivLoss(reduction="batchmean")
#             ICCS = torch.empty((128, 128)).cuda()
#             ICCT = torch.empty((128, 128)).cuda()
#             for i in range(128):
#                 for j in range(i, 128):
#                     ICCS[j, i] = ICCS[i, j] = kl(s_logit[:, i], s_logit[:, j])
#                     ICCT[j, i] = ICCT[i, j] = kl(t_logit[:, i], t_logit[:, j])

#             ICCS = torch.nn.functional.normalize(ICCS, dim=1)
#             ICCT = torch.nn.functional.normalize(ICCT, dim=1)
#             # lo_loss = self.lo_lambda * (ICCS - ICCT).pow(2).mean() / b
#             loss1 = (ICCS - ICCT).pow(2)
#             lo_loss = self.lo_lambda * loss1.sum()
        return pi_loss, lo_loss


def get_AICSD_model(batchsize):
    model = AICSD_model(batchsize)
    device = 'cuda'
    model.to(device)
    return model


if __name__ == '__main__':
    model = get_AICSD_model(batchsize=8)

    # gt = torch.ones(2, 1, 512, 1024).cuda()
    f_t = torch.randn(4, 768, 128, 128).cuda()
    # f_t=torch.randn(2,1, 128,256).cuda()
    f_s = torch.randn(4, 128, 64, 64).cuda()
    # gt=torch.ones(8, 1, 512, 1024).cuda()#B,512,1024
    t_logit = torch.randn(4,19,128,128).cuda()
    s_logit = torch.randn(4,19,64,64).cuda()
    output,output1 = model(f_s, f_t, t_logit, s_logit)
    # 输入fs,fc,gt返回cam和处理过的gt
    print("Number of Loss1:", output)
    print("Number of Loss2:", output1)
    # loss_seg = self.criterion(output, target)
    ########### uncomment lines below for ALW ##################
    # alpha = epoch/120
    # loss = alpha * (loss_seg + lo_loss) + (1-alpha) * pi_loss