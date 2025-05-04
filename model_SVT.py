import torch.nn as nn
import torch.nn.functional as F

class LocallyGroupedSelfAttention(nn.Module):
    """Locally-grouped Self-Attention"""
    def __init__(self, dim, num_heads, window_size=(5, 1), attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size  # 确保这是一个元组
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = 200  # 修改为合适的高度
        W = 1    # 修改为合适的宽度
        h_group, w_group = H // self.window_size[0], W // self.window_size[1]

        # 重塑张量
        x = x.view(B, h_group, self.window_size[0], w_group, self.window_size[1], C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * h_group * w_group, self.window_size[0] * self.window_size[1], C)

        # 计算 Q, K, V
        qkv = self.qkv(x).view(B * h_group * w_group, self.window_size[0] * self.window_size[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B * h_group * w_group, self.window_size[0] * self.window_size[1], C)
        x = x.view(B, h_group, w_group, self.window_size[0], self.window_size[1], C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, N, C)

        # 投影层
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalSubsampledAttention(nn.Module):
    """Global Sub-sampled Attention"""
    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: tuple):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with LSA and GSA"""
    def __init__(self, dim, num_heads, window_size=(5, 1), sr_ratio=1, attn_drop=0., proj_drop=0., mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lsa = LocallyGroupedSelfAttention(dim, num_heads, window_size, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_drop)
        )
        self.gsa = GlobalSubsampledAttention(dim, num_heads, sr_ratio, attn_drop, proj_drop)

    def forward(self, x, size: tuple):
        lsa_out = x + self.lsa(self.norm1(x))
        lsa_mlp = lsa_out + self.mlp(self.norm2(lsa_out))
        gsa_out = lsa_mlp + self.gsa(self.norm1(lsa_mlp), size)
        return gsa_out

class TwinsSVT(nn.Module):
    """Twins-SVT Model"""
    def __init__(self, input_dim=5, embed_dim=128, num_heads=8, num_layers=4, output_dim=320, window_size=(5, 1), sr_ratio=1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, embed_dim)

        # Transformer 编码器
        self.transformer_block = TransformerBlock(embed_dim, num_heads, window_size, sr_ratio)
        
        # 输出层
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, embed_dim)

        # Transformer 编码
        H, W = int(x.shape[1] ** 0.5), x.shape[1] // int(x.shape[1] ** 0.5)
        x = self.transformer_block(x, (H, W))

        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # 输出层
        x = self.output_layer(x)  # (batch_size, output_dim)
        
        return x
    
# # 示例用法
# if __name__ == "__main__":
#     # 输入数据
#     input_data = torch.randn([10, 200, 5])  # [batch_size, height, width, channels]
#     model = TransformerModel()
#     output = model(input_data)
#     print(output.shape)