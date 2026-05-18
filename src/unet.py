# unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Conv + GroupNorm + SiLU"""
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if time_dim is not None:
            self.time_mlp = nn.Linear(time_dim, out_ch)
        else:
            self.time_mlp = None
        
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)[:, :, None, None]
            h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x, t)
        return self.pool(x), x  # pooled, skip connection


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t)


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for DDPM
    Conditions: treatment (a), time delta (hours_diff)
    """
    def __init__(self, in_channels=1, base_channels=64, time_dim=256):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Diffusion timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Treatment embedding (a: 0 or 1)
        self.treatment_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Delta embedding (hours_diff: continuous)
        self.delta_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Input: y_t (1ch) + x_0 (1ch) = 2ch
        self.input_channels = in_channels * 2
        
        # Encoder
        ch = base_channels
        self.enc1 = DownBlock(self.input_channels, ch, time_dim)      # 2 -> 64
        self.enc2 = DownBlock(ch, ch * 2, time_dim)                   # 64 -> 128
        self.enc3 = DownBlock(ch * 2, ch * 4, time_dim)               # 128 -> 256
        self.enc4 = DownBlock(ch * 4, ch * 8, time_dim)               # 256 -> 512
        
        # Bottleneck
        self.bottleneck = ConvBlock(ch * 8, ch * 8, time_dim)         # 512 -> 512
        
        # Decoder (in_ch, skip_ch, out_ch)
        self.dec4 = UpBlock(ch * 8, ch * 8, ch * 4, time_dim)         # 512, skip=512 -> 256
        self.dec3 = UpBlock(ch * 4, ch * 4, ch * 2, time_dim)         # 256, skip=256 -> 128
        self.dec2 = UpBlock(ch * 2, ch * 2, ch, time_dim)             # 128, skip=128 -> 64
        self.dec1 = UpBlock(ch, ch, ch, time_dim)                     # 64, skip=64 -> 64
        
        # Output
        self.out = nn.Conv2d(ch, in_channels, 1)

    def forward(self, y_t, t, x_0, a, delta):
        """
        Args:
            y_t: [B, 1, H, W] noisy follow-up
            t: [B] diffusion timestep
            x_0: [B, 1, H, W] baseline image
            a: [B, 1] treatment (0 or 1)
            delta: [B, 1] hours difference (normalized)
        """
        # Combine all conditions
        t_emb = self.time_mlp(t)
        a_emb = self.treatment_mlp(a)
        delta_emb = self.delta_mlp(delta)
        cond = t_emb + a_emb + delta_emb
        
        # Concatenate y_t and x_0
        x = torch.cat([y_t, x_0], dim=1)
        
        # Encoder
        x, skip1 = self.enc1(x, cond)
        x, skip2 = self.enc2(x, cond)
        x, skip3 = self.enc3(x, cond)
        x, skip4 = self.enc4(x, cond)
        
        # Bottleneck
        x = self.bottleneck(x, cond)
        
        # Decoder
        x = self.dec4(x, skip4, cond)
        x = self.dec3(x, skip3, cond)
        x = self.dec2(x, skip2, cond)
        x = self.dec1(x, skip1, cond)
        
        return self.out(x)


# 테스트
if __name__ == '__main__':
    model = ConditionalUNet(in_channels=1, base_channels=64)
    
    B = 4
    y_t = torch.randn(B, 1, 128, 128)
    t = torch.randint(0, 1000, (B,))
    x_0 = torch.randn(B, 1, 128, 128)
    a = torch.randint(0, 2, (B, 1)).float()
    delta = torch.rand(B, 1) * 48  # 0-48 hours
    
    noise_pred = model(y_t, t, x_0, a, delta)
    print(f"Input shape: {y_t.shape}")
    print(f"Output shape: {noise_pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")