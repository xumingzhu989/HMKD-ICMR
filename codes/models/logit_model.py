
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['get_logit_model']




class Logit_model(nn.Module):
    def __init__(self,opt,num_classes,in_chans):
        super(Logit_model, self).__init__()
        self.opt=opt
        self.num_classes=num_classes
        self.in_chans=in_chans
        # #fs先通过卷积投影到ft_final空间，后通过分割头，得到p_fs<-->pt
        # fs先通过卷积1，后通过分割头，得到p_fs<-->pt
        self.proj=nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ).to('cuda') for in_ch in in_chans])
        self.pred_head = nn.ModuleList([nn.Conv2d(64, self.num_classes, kernel_size=1).to('cuda') for in_ch in in_chans])
        self.criterionLogit=nn.MSELoss()
        # self.criterionLogit = nn.KLDivLoss(reduction='batchmean')




    def forward(self,pt,fs):
        loss=[]
        #alpha=[0.1,0.1,0.1,0.1,1]
        for i in range(len(self.opt)):
            if i !=0:
                #fs调整到和pt的输入特征一样大
                _,_,ht,wt=pt.size()
                fs[i] = F.interpolate(fs[i], (ht, wt), mode='bilinear', align_corners=True)
            fs[i] = self.proj[i](fs[i])
            #预测
            fs[i]=self.pred_head[i](fs[i])
            #计算分割图的MSE
            #loss.append(self.criterionLogit(fs[i], pt).to('cuda'))
            loss.append(self.criterionLogit(F.log_softmax(fs[i], dim=-1), F.softmax(pt, dim=-1)).to('cuda'))
            # loss.append(alpha[i] * self.criterionLogit(F.log_softmax(fs[i], dim=-1), F.softmax(pt, dim=-1)).to('cuda'))


        return loss




def get_logit_model(opt, num_classes, in_chans):
    model = Logit_model(opt, num_classes, in_chans)  # Create an instance of Logit_model
    device = 'cuda'
    model.to(device)
    return model



if __name__ == '__main__':
    model = get_logit_model(opt=[0,1,2,3],num_classes=19,in_chans=[64, 128, 256, 512])
    #输入pt ps (opt : fs_1 fs_2 fs_3 fs_4 fs_a ps)
    '''
    1 2 3 4 5 6
    cnn_c1.shape	torch.Size([2, 64, 128, 256])
    cnn_c2.shape	torch.Size([2, 128, 64, 128])
    cnn_c3.shape	torch.Size([2, 256, 64, 128])
    cnn_c4.shape	torch.Size([2, 512, 64, 128])
    cnn_aspp.shape	torch.Size([2, 128, 64, 128])
    ps 2, 19, 64, 128
    
    pt 2, 19, 128, 256
    ps 2, 19, 64, 128
    
    self.linear_pred = nn.Conv2d(decoder_dim=768, num_classes=9, kernel_size=1)
    
    
    '''
    pt = torch.randn(4, 19, 128, 128).cuda()
    fs_1 = torch.randn(4, 64, 128, 128).cuda()
    fs_2 = torch.randn(4, 128, 64, 64).cuda()
    fs_3 = torch.randn(4, 256, 64, 64).cuda()
    fs_4 = torch.randn(4, 512, 64, 64).cuda()
    # fs_a = torch.randn(2, 128, 64, 128).cuda()
    # ps = torch.randn(2, 19, 64, 128).cuda()
    fs=[fs_1,fs_2,fs_3,fs_4]
    output = model(pt,fs)
    ##输入pt ps fs_a
    #output = model()

