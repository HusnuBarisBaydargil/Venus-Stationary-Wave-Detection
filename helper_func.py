import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class LayerNorm(nn.Module):
    def __init__(self, channels, height, width):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=[channels, height, width], eps=1e-6, elementwise_affine=True)

    def forward(self, x):
        return self.ln(x)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.up_conv(x)
    
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.q(x).view(b, c, h * w).permute(0, 2, 1)
        k = self.k(x).view(b, c, h * w)
        v = self.v(x).view(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn * (c ** -0.5), dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.proj_out(out)
