import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CXRLatentAutoencoder(nn.Module):
    """Small KL autoencoder for 1-channel CXR images normalized to [-1, 1]."""

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 4),
    ):
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]

        encoder_layers = [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            encoder_layers.append(ResidualBlock(in_ch, out_ch))
            encoder_layers.append(ResidualBlock(out_ch, out_ch))
            if level < len(channels) - 1:
                encoder_layers.append(Downsample(out_ch))
            in_ch = out_ch
        encoder_layers.extend(
            [
                nn.GroupNorm(_group_count(in_ch), in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, 2 * latent_channels, kernel_size=3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1),
            ResidualBlock(channels[-1], channels[-1]),
        ]
        in_ch = channels[-1]
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            decoder_layers.append(ResidualBlock(in_ch, out_ch))
            decoder_layers.append(ResidualBlock(out_ch, out_ch))
            if level > 0:
                decoder_layers.append(Upsample(out_ch))
            in_ch = out_ch
        decoder_layers.extend(
            [
                nn.GroupNorm(_group_count(in_ch), in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode_moments(self, image):
        moments = self.encoder(image)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def encode(self, image, sample=False):
        mu, logvar = self.encode_moments(image)
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, image, sample=True):
        mu, logvar = self.encode_moments(image)
        if sample:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar


def kl_divergence(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
