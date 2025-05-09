"""Pyramid Scene Parsing Network"""
import math
import sys
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get_GLA_model']


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=8, stride=8, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=0)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x) # torch.Size([4, 256, 64])
        return x, H, W#打印x的维度！

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, patches):
        # print(patches.shape)
        # patches: (batch_size, num_patches, embed_size)
        # 线性变换得到 Q, K, V
        Q = self.query(patches)  # (batch_size, num_patches, embed_size)
        K = self.key(patches)     # (batch_size, num_patches, embed_size)
        V = self.value(patches)   # (batch_size, num_patches, embed_size)
        # 计算注意力分数 (Q * K^T) / sqrt(d_k)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, num_patches, num_patches)
        d_k = Q.size(-1)
        attention_scores /= d_k ** 0.5  # 缩放
        
        # 计算权重（使用 softmax）
        attention_weights = self.softmax(attention_scores)  # (batch_size, num_patches, num_patches)
        # 计算加权求和得到输出
        output = torch.bmm(attention_weights, V)  # (batch_size, num_patches, embed_size)
        # print('output:',output.shape)
        # print('attention:',attention_weights.shape)
        return output, attention_weights #torch.Size([4, 256, 64]) torch.Size([4, 256, 256])


class GLA_model(nn.Module):
    def __init__(self, batchsize):
        super(GLA_model, self).__init__()
        self.batchsize = batchsize
        self.criterion = nn.MSELoss()

        '''
        stage1
        torch.Size([4, 16384, 64])  B 128 128 C
        torch.Size([4, 64, 128, 128])
        
        stage2
        torch.Size([4, 4096, 128])  B 64 64 C
        torch.Size([4, 128, 64, 64])
        
        stage3
        torch.Size([4, 1024, 320])  B 32 32 C
        torch.Size([4, 256, 64, 64])
        
        stage4
        torch.Size([4, 256, 512]) B 16 16 C
        torch.Size([4, 512, 64, 64])
        
        vit patch embed
        stage 1 2 3 4
        stride 4 2 2 2
        patch size 7 3 3 3
        
        cnn patch embed
        stage 1 2 3 4
        stride 1 1 2 2
        patch size 7 3 3 3

        '''
        img_size=[512,512]
        embed_dims=[128,128,256,320,512,512,64,64]

        # self.sim=torch.nn.CosineSimilarity(dim=-1)
        self.sim=torch.nn.CosineSimilarity(dim=1)
        self.patch_embed_s4 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=4,
            stride=4,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.patch_embed_t4 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=4,
            stride=4,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.sim=torch.nn.CosineSimilarity(dim=1)
        self.patch_embed_s8 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=8,
            stride=8,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.patch_embed_t8 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=8,
            stride=8,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.sim=torch.nn.CosineSimilarity(dim=1)
        self.patch_embed_s16 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=16,
            stride=16,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.patch_embed_t16 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=16,
            stride=16,
            in_chans=embed_dims[-2],
            embed_dim=embed_dims[-1]
        )
        self.patch_embed_s8_04 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=8,
            stride=8,
            in_chans=32,
            embed_dim=32
        )
        self.patch_embed_t8_04 = PatchEmbed(
            img_size=(img_size[0], img_size[1]),
            patch_size=8,
            stride=8,
            in_chans=32,
            embed_dim=32
        )
        self.sa_1 = SelfAttention(embed_dims[-1])
        self.sa_2 = SelfAttention(embed_dims[-1])
        self.sa_1_04 = SelfAttention(32)
        self.sa_2_04 = SelfAttention(32)
        self.conv_c1010=nn.Conv2d(256, 32, kernel_size=1)
    def forward(self, f_t, f_s):
#         B,C,H,W = f_s .shape
#         f_t = f_t.reshape(B,C,H,W) # torch.Size([4, 64, 128, 128])
#         # #4*4
#         # f_t4, H, W = self.patch_embed_t4(f_t) # torch.Size([4, 1024, 64]) 
#         # f_s4, H, W = self.patch_embed_s4(f_s) # torch.Size([4, 1024, 64])
#         # ca_s_14,ca_s4 = self.sa_1(f_s4) # torch.Size([4, 1024, 64]) torch.Size([4, 1024, 1024])
#         # ca_t_14,ca_t4 = self.sa_2(f_t4)
#         # loss1 = (ca_t4 - ca_s4)**2
#         # loss14 = loss1.sum()
#         #8*8
#         f_t8, H, W = self.patch_embed_t8(f_t) # torch.Size([4, 256, 64]) 
#         f_s8, H, W = self.patch_embed_s8(f_s) # torch.Size([4, 256, 64])
#         ca_s_18,ca_s8 = self.sa_1(f_s8) # torch.Size([4, 256, 64]) torch.Size([4, 256, 256])
#         ca_t_18,ca_t8 = self.sa_2(f_t8)
#         loss2 = (ca_t8 - ca_s8)**2
#         loss28 = loss2.sum()
#         # #16*16
#         # f_t16, H, W = self.patch_embed_t16(f_t) # torch.Size([4, 64, 64]) 
#         # # print("f_t16:",f_t16.shape)
#         # f_s16, H, W = self.patch_embed_s16(f_s) # torch.Size([4, 64, 64])
#         # # print("f_s16:",f_s16.shape)
#         # ca_s_116,ca_s16 = self.sa_1(f_s16) # torch.Size([4, 64, 64]) torch.Size([4, 64, 64])
#         # # print("ca_s_116:",ca_s_116.shape)
#         # # print("ca_s16:",ca_s16.shape)
#         # ca_t_116,ca_t16 = self.sa_2(f_t16)
#         # # print("ca_t_116:",ca_t_116.shape)
#         # # print("ca_t16:",ca_t16.shape)
#         # # loss = 1 - torch.mean(self.sim(ca_s, ca_t))
#         # # loss = loss1.sum()
#         # # loss_st = self.criterion(ca_t,ca_s)
#         # loss3 = (ca_t16 - ca_s16)**2
#         # loss316 = loss3.sum()
        
#         # loss = loss14 + loss28 + loss316
        # print(f_t.shape)
        # print(f_s.shape)
        # torch.Size([4, 256, 90, 120])
        # torch.Size([4, 10800, 32])
        B=4
        C2=32
        H1=90
        W1=120
        # B,C,H,W = f_s .shape
        # print(f_t.shape) # torch.Size([4, 16384, 32])
        f_s = f_s.reshape(B,C2,H1,W1)# torch.Size([4, 32, 128, 128])
        f_t = self.conv_c1010(f_t)
        f_t8, H, W = self.patch_embed_t8_04(f_t) # torch.Size([4, 256, 64]) 
        f_s8, H, W = self.patch_embed_s8_04(f_s) # torch.Size([4, 256, 64])
        ca_s_18,ca_s8 = self.sa_1_04(f_s8) # torch.Size([4, 256, 64]) torch.Size([4, 256, 256])
        ca_t_18,ca_t8 = self.sa_2_04(f_t8)
        loss2 = (ca_t8 - ca_s8)**2
        loss28 = loss2.sum()
        
        return loss28



def get_GLA_model(batchsize):
    model = GLA_model(batchsize)
    device = 'cuda'
    model.to(device)
    return model


if __name__ == '__main__':


    model = get_GLA_model(batchsize=4)
    # #stage 2 3 4
    # f_t = [torch.randn(4, 4096, 128).cuda(), torch.randn(4, 4096, 128).cuda(), torch.randn(4, 1024, 320).cuda(), torch.randn(4, 1024, 320).cuda(),torch.randn(4, 256, 512).cuda(),torch.randn(4, 256, 512).cuda()]
    # f_s = [torch.randn(4, 128, 64, 64).cuda(), torch.randn(4, 256, 64, 64).cuda(), torch.randn(4, 512, 64, 64).cuda()]

    #stage 1 2 3 4
    f_t = [torch.randn(4, 4096, 128).cuda(), torch.randn(4, 1024, 320).cuda(), torch.randn(4, 256, 512).cuda(),torch.randn(4, 16384, 64).cuda()]
    f_s = [torch.randn(4, 128, 64, 64).cuda(), torch.randn(4, 256, 64, 64).cuda(), torch.randn(4, 512, 64, 64).cuda(),torch.randn(4, 64, 128, 128).cuda()]

    output = model(f_t, f_s)
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    print("Number of parameter: %.2fM" %(total / 1e6))

    # 输入fs,fc,gt返回cam和处理过的gt
