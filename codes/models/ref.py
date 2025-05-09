class CannyConv_H(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(CannyConv_H, self).__init__()
        # Canny边缘检测的卷积核 (这里简化为一个水平方向的Sobel算子)
        self.canny_kernel = torch.tensor([[-1., 0., 1.],
                                          [-2., 0., 2.],
                                          [-1., 0., 1.]], dtype=torch.float32)
        # 扩展卷积核以匹配输入和输出的通道数
        self.canny_kernel = self.canny_kernel.repeat(out_channels, in_channels, 1, 1)

        # 创建卷积层，但不添加偏置项且不使用默认的权重初始化
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 手动设置卷积核权重
        with torch.no_grad():
            self.conv.weight.copy_(self.canny_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CannyConv_V(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(CannyConv_V, self).__init__()
        # Canny边缘检测的卷积核 (这里简化为一个垂直方向的Sobel算子)
        self.canny_kernel = torch.tensor([[1., 2., 1.],
                                          [0., 0., 0.],
                                          [-1., -2., -1.]], dtype=torch.float32)
        # 扩展卷积核以匹配输入和输出的通道数
        self.canny_kernel = self.canny_kernel.repeat(out_channels, in_channels, 1, 1)

        # 创建卷积层，但不添加偏置项且不使用默认的权重初始化
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 手动设置卷积核权重
        with torch.no_grad():
            self.conv.weight.copy_(self.canny_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x