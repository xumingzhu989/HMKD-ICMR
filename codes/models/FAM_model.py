"""Pyramid Scene Parsing Network"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist

__all__ = ['get_FAM_model']

class FAM_model(nn.Module):
    def __init__(self, batchsize = 2, in_channels = 64, out_channels = 64, shapes = 128, kernel_size = 3, stride = 1, padding = 1, groups = 4, bias = False):
        super(FAM_model, self).__init__()
        self.batchsize = batchsize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes  # shapes指定了输入、输出特征图的高度和宽度（即 (height, width)）
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # 断言：确保输出通道数可以被分组数整除，这是分组卷积的一个要求
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # self.rel_h和self.rel_w是两个可学习的参数，用于向模型中引入相对位置信息，它们分别对应高度和宽度方向上的位置编码
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        # 定义了三个1x1的卷积层（key_conv, query_conv, value_conv），用于从输入中提取不同的特征表示，这些特征将用于后续的注意力计算
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # 调用reset_parameters方法
        self.reset_parameters()

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.scale = (1 / (self.in_channels * self.out_channels))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, 128, 128, dtype=torch.cfloat))
        self.weights1_real = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes))#128,128
        self.weights1_imag = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes))#128,128
        #print(self.weights1)
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.con_vv = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.con_vv1 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)


    def compl_mul2d(self, input_real, input_imag, weights_real, weights_imag):
        weights_real = weights_real.permute(0, 2, 3, 1) 
        weights_imag = weights_imag.permute(0, 2, 3, 1)
        real_part = torch.einsum("bchw,ohwi->bochw", input_real, weights_real) - torch.einsum("bchw,ohwi->bochw", input_imag, weights_imag)
        imag_part = torch.einsum("bchw,ohwi->bochw", input_real, weights_imag) + torch.einsum("bchw,ohwi->bochw", input_imag, weights_real)
        # Reshape the output to desired format
        return real_part.view(input_real.size(0), -1, input_real.size(3)), \
               imag_part.view(input_imag.size(0), -1, input_imag.size(3))


    # 初始化卷积层和位置编码参数的权重
    def reset_parameters(self):
        # 使用 Kaiming 初始化方法来初始化卷积层的权重，这是一种针对 ReLU 激活函数的常用初始化方法，有助于保持输入和输出的方差一致
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        # 对于位置编码参数 self.rel_h 和 self.rel_w，则使用标准正态分布进行初始化
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


    def forward(self, fs, ft):
        #print(fs.size())
        # 获取输入 fs 的尺寸（批量大小、通道数、高度、宽度）
        batch, channels, height, width = fs.size()
        
        # 对 fs 进行填充
        padded_fs = F.pad(fs, [self.padding, self.padding, self.padding, self.padding])
        # 通过 key_conv, query_conv, value_conv 对输入进行卷积，以生成键（key）、查询（query）和值（value）特征图
        q_out = self.query_conv(fs)
        k_out = self.key_conv(padded_fs)
        v_out = self.value_conv(padded_fs)

        # 使用 unfold 方法将 k_out 和 v_out 沿着高度和宽度方向展开，以模拟局部窗口的注意力计算。每个窗口的大小由 kernel_size 和 stride 确定
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # 将 k_out 分成两部分（k_out_h 和 k_out_w），分别对应于高度和宽度方向的特征
        # 然后，将相对位置编码 self.rel_h 和 self.rel_w 加到对应的部分上，以增强模型对位置信息的感知能力
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        # 将 k_out 和 v_out 重塑为适合后续注意力计算的形状，并同时重塑 q_out 以匹配 k_out 的维度
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        # 计算 q_out 和 k_out 的点积，并通过 softmax 函数在最后一个维度（即窗口内的元素）上应用归一化，以生成注意力权重
        # 使用这些权重对 v_out 进行加权求和，得到最终的注意力特征图
        # 将加权求和的结果重塑回原始的空间维度（高度和宽度），并返回
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # print(out.size())
        # return out
        if isinstance(out, tuple):
            out, cuton = out
        else:
            cuton = 0.1
        batchsize = out.shape[0]
        # print(out.shape)
        fs_ft = torch.fft.fft2(out, norm="ortho")
        # out_ft = self.compl_mul2d(fs_ft, self.weights1)
        fs_ft_real = fs_ft.real
        fs_ft_imag = fs_ft.imag

        out_ft_real, out_ft_imag = self.compl_mul2d(fs_ft_real, fs_ft_imag, self.weights1_real, self.weights1_imag)

        out_ft = torch.complex(out_ft_real, out_ft_imag)
        # print("  **  ")
        # print(out_ft.dtype)
        # print("  **  ")
        # out_ft = 
        batch_fftshift = batch_fftshift2d(out_ft)

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.
        ##
        #print(batch_fftshift.shape)
        batch_size = batch_fftshift.shape[0]
        channels = self.in_channels#64
        width = batch_fftshift.shape[2]
        height = batch_fftshift.numel() // (batch_size * channels * width * 2)  # Calculate height
        batch_fftshift = batch_fftshift.view(batch_size, channels, height, width, 2)
        #print(batch_fftshift.shape)
        ##
        batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
        # test with batch shift
        out_ft = batch_ifftshift2d(batch_fftshift)  # 执行IFFT shift
        out_ft = torch.view_as_complex(out_ft)
        # Return to physical space
        out1 = torch.fft.ifft2(out_ft, s=(out.size(-2), out.size(-1)), norm="ortho").real
        out = self.con_vv(out)
        out2 = self.w0(out)  # 通过 self.w0 对原始输入 x 执行 1x1 卷积，得到 out2
        out2 = self.con_vv1(out2)
        # print(out1.shape)
        # print(out2.shape)
        make_fs = self.rate1 * out1 + self.rate2 * out2 # torch.Size([1, 256, 64, 64])
        # print(make_fs.size())
        # print(ft.size)
        ##
        # print(ft.shape)
        
        if self.out_channels==320:
            batch = ft.shape[0]  # 2
            height = 32
            width = 32
            channels = self.out_channels # 64
            ft = ft.view(batch,channels, height, width)
            ft = F.interpolate(input=ft, size=(64, 64), mode='bilinear', align_corners=True)
            ft=self.con_vv(ft)
            ft = ft.permute(0, 2, 3, 1)
        else:
            batch = ft.shape[0]  # 2
            height = self.shapes #128
            width = self.shapes #128
            channels = self.out_channels # 64
            ft = ft.view(batch, height, width, channels)
        ##
        # ft = ft.view(batch, height, width, channels).permute(0,3,1,2)
        # print("Shape of make_fs:", make_fs.shape)
        # print("Shape of ft:", ft.shape)
        make_fs = make_fs.permute(0, 2, 3, 1) # Swap dimensions to match ft's shape
        loss1 = F.mse_loss(make_fs, ft, reduction="mean") # 计算当前特征图对 fs 和 ft 之间的均方误差（MSE）损失，将所有元素的损失平均化
        loss = loss1.item() ##
        return loss  # 将来自FFT处理的输出 out 和来自卷积处理的输出 out2 通过 rate1 和 rate2 加权求和，返回最终结果


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


def batch_fftshift2d(x):
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None)
                  if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
                  if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def get_FAM_model(batchsize, in_channels, out_channels, shapes, kernel_size = 3, stride = 1, padding = 1, groups = 4, bias = False):
    model = FAM_model(batchsize,in_channels, out_channels, shapes, kernel_size, stride, padding, groups, bias)
    device = 'cuda'
    model.to(device)
    return model

if __name__ == '__main__':
    model = get_FAM_model(batchsize=2, in_channels=64, out_channels=64, shapes=128, kernel_size = 3, stride = 1, padding = 1, groups = 4, bias = False)
    # model = get_FAM_model(batchsize=2, in_channels=256, out_channels=320, shapes=64, kernel_size = 3, stride = 1, padding = 1, groups = 4, bias = False)
    #((128,128),(128,128))
    
    f_t = torch.randn(2, 16384, 64).cuda()
    f_s = torch.randn(2, 64, 128, 128).cuda()
    # f_t = torch.randn(1, 1024, 320).cuda() #t_outputs: torch.Size([4, 1024, 320])
    # f_s = torch.randn(1, 256, 64, 64).cuda()
    output = model(f_s, f_t)
    # 输入fs,fc,gt返回cam和处理过的gt
    print("Number of Loss:",output)