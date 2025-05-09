import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

__all__ = ['get_DTKD_model']


class DTKD_model(nn.Module):
    def __init__(self, batchsize, ALPHA=3.0,BETA=1.0,T=4,WARMUP=20):
        super(DTKD_model, self).__init__()
        self.batchsize = batchsize
        self.alpha = ALPHA
        self.beta = BETA
        self.warmup = WARMUP
        self.temperature = T
        self.fc = nn.Linear(19, 19)

    def forward(self, logits_s, logits_t, iteration):
        # DTKD Loss
        B,C,H,W = logits_s.shape
        logits_t = F.interpolate(input=logits_t, size=(H, W), mode='bilinear', align_corners=True)
        reference_temp = self.temperature
        logits_s_max, _ = logits_s.max(dim=1, keepdim=True)
        logits_t_max, _ = logits_t.max(dim=1, keepdim=True)
        logits_s_temp = 2 * logits_s_max / (logits_t_max + logits_s_max) * reference_temp 
        logits_t_temp = 2 * logits_t_max / (logits_t_max + logits_s_max) * reference_temp
        
        ourskd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_s / logits_s_temp, dim=1) , # 学生
            F.softmax(logits_t / logits_t_temp, dim=1)       # 老师
        ) 
        loss_ourskd = (ourskd.sum(1, keepdim=True) * logits_t_temp * logits_s_temp).mean()
        
        # Vanilla KD Loss
        vanilla_temp = self.temperature
        kd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_s / vanilla_temp, dim=1) , # 学生
            F.softmax(logits_t / vanilla_temp, dim=1)       # 老师
        ) 
        loss_kd = (kd.sum(1, keepdim=True) * vanilla_temp ** 2).mean() 
         
        loss_dtkd = min(iteration / self.warmup, 1.0) * (self.alpha * loss_ourskd + self.beta * loss_kd)
        return  loss_dtkd*0.01
def get_DTKD_model(batchsize=4,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20):
    model = DTKD_model(batchsize,ALPHA,BETA,T,WARMUP)
    device = 'cuda'
    model.to(device)
    return model

if __name__ == '__main__':
    model = get_DTKD_model(batchsize=4,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20)

    logit_t = torch.randn(4,19,128,128).cuda()
    logit_s = torch.randn(4,19,64,64).cuda()
    iteration = 10
    output = model(logit_s,logit_t,iteration)
    print("Number of Loss:",output)

# ////
# # DTKD(Dynamic Temperature Knowledge Distillation) CFG
# CFG.DTKD = CN()
# CFG.DTKD.LOSS = CN()
# CFG.DTKD.ALPHA = 3.0      # DTKD
# CFG.DTKD.BETA = 1.0       # KD
# CFG.DTKD.CE_WEIGHT = 1.0  # CE
# CFG.DTKD.T = 4
# CFG.DTKD.WARMUP = 20
