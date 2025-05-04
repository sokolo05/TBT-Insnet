import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import math
import warnings
from loss_function import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ECALayer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = self._compute_kernel_size()
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, 
                            padding=(self.kernel_size-1)//2, 
                            bias=False)
        
    def _compute_kernel_size(self):
        k = int(abs((math.log(self.in_channel, 2) + 1) / 2))
        return k if k % 2 != 0 else k + 1  # 确保奇数的更可靠方法
        
    def forward(self, x):
        # 维度验证
        if x.dim() != 4:
            raise ValueError(f"输入维度错误：期望4D输入，得到{x.dim()}D张量")
        
        # 通道数验证
        B, C, H, W = x.shape
        if C != self.in_channel:
            raise ValueError(f"通道数不匹配：初始化通道数{self.in_channel}，实际输入{C}")

        # 特征压缩
        y = x.mean(dim=(2, 3))  # [B, C]
        
        # 1D卷积处理
        y = y.unsqueeze(1)     # [B, 1, C]
        y = self.conv(y)       # [B, 1, C]
        
        # 激活与维度扩展
        y = torch.sigmoid(y)
        y = y.view(B, C, 1, 1)  # 更直观的维度变换
        
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super().__init__()
        self.filters = max(1, in_channel // ratio)
        self.shared_layer = nn.Sequential(
            nn.Linear(in_channel, self.filters, bias=True),
            nn.ReLU(),
            nn.Linear(self.filters, in_channel, bias=True)
        )
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        avg_pool = x.mean(dim=(2,3))  # [B, C]
        max_pool = x.amax(dim=(2,3))  # [B, C]
        
        avg_out = self.shared_layer(avg_pool)
        max_out = self.shared_layer(max_pool)
        
        y = torch.sigmoid(avg_out + max_out)
        return x * y.unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=(1, 5)):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                            padding=((kernel_size[0]-1)//2, (kernel_size[1]-1)//2), 
                            bias=False)
        
    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool = x.amax(dim=1, keepdim=True)  # [B, 1, H, W]
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        y = torch.sigmoid(self.conv(concat))
        return x * y

class CBAMBlock(nn.Module):
    def __init__(self, in_channel, ratio=8, kernel_size=(1,5)):
        super().__init__()
        self.channel_att = ChannelAttention(in_channel, ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class Insnet_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            SeparableConv2d(128, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            CBAMBlock(64, ratio=7, kernel_size=(1,5)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            ECALayer(64),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            ECALayer(64),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2,1)),
            
            nn.Flatten()
        )
        
    def forward(self, x):
        
        # print(f"输入: {x.shape}")  # 应为 [B*T, 1, 200, 5]
        x = self.layers[0](x)  # 第一个Conv2d层
        # print(f"Conv1 输出: {x.shape}")  # 应为 [B*T, 128, 200, 5]
        
        x = self.layers[1](x)  # ELU激活
        x = self.layers[2](x)  # MaxPool
        # print(f"MaxPool1 后: {x.shape}")  # 应为 [B*T, 128, 100, 5]
        
        x = self.layers[3](x)  # 第一个深度可分离卷积
        # print(f"SepConv1 输出: {x.shape}")  # 应为 [B*T, 64, 100, 5]
        
        x = self.layers[4](x)  # ELU激活
        x = self.layers[5](x)  # MaxPool
        # print(f"MaxPool2 后: {x.shape}")  # 应为 [B*T, 64, 50, 5]
        
        x = self.layers[6](x)  # CBAM块
        # print(f"CBAMBlock 输出: {x.shape}")  # 应为 [B*T, 64, 50, 5]
        
        x = self.layers[7](x)  # 第二个深度可分离卷积
        # print(f"SepConv2 输出: {x.shape}")  # 应为 [B*T, 64, 50, 5]
        
        x = self.layers[8](x)  # ELU激活
        x = self.layers[9](x)  # MaxPool
        # print(f"MaxPool3 后: {x.shape}")  # 应为 [B*T, 64, 25, 5]
        
        x = self.layers[10](x)  # 第三个深度可分离卷积
        # print(f"SepConv3 输出: {x.shape}")  # 应为 [B*T, 64, 25, 5]
        
        x = self.layers[11](x)  # ELU激活
        x = self.layers[12](x)  # MaxPool
        # print(f"MaxPool4 后: {x.shape}")  # 应为 [B*T, 64, 12, 5]
        
        x= self.layers[13](x)  # ECALayer
        # print(f"ECALayer 1 输出: {x.shape}")  # 应为 [B*T, 64, 3, 5]
        
        x = self.layers[14](x)  # 第四个深度可分离卷积
        # print(f"SepConv4 输出: {x.shape}")  # 应为 [B*T, 64, 12, 5]
        
        x = self.layers[15](x)  # ELU激活
        x = self.layers[16](x)  # MaxPool
        # print(f"MaxPool5 后: {x.shape}")  # 应为 [B*T, 64, 6, 5]
        
        x = self.layers[17](x)  # 第五个深度可分离卷积
        # print(f"SepConv5 输出: {x.shape}")  # 应为 [B*T, 64, 6, 5]
        
        x = self.layers[18](x)  # ELU激活
        x = self.layers[19](x)  # MaxPool
        # print(f"MaxPool6 后: {x.shape}")  # 应为 [B*T, 64, 3, 5]
        
        x= self.layers[20](x)  # ECALayer
        # print(f"ECALayer 2 输出: {x.shape}")  # 应为 [B*T, 64, 3, 5]
        
        x = self.layers[21](x)  # 第六个深度可分离卷积
        # print(f"SepConv6 输出: {x.shape}")  # 应为 [B*T, 64, 3, 5]
        
        x = self.layers[22](x)  # ELU激活
        x = self.layers[23](x)  # MaxPool
        # print(f"MaxPool7 后: {x.shape}")  # 应为 [B*T, 64, 1, 5]
        
        x = self.layers[24](x)  # Flatten
        # print(f"Flatten 输出: {x.shape}")  # 应为 [B, T, 320]
        
        return x
    

# # 示例用法
# if __name__ == "__main__":
#     # 输入数据
#     input_data = torch.randn([10, 1, 200, 5])  # [batch_size, time_steps, height, width, channels]
#     model = Insnet_model()
#     output = model(input_data)
#     print(output.shape)