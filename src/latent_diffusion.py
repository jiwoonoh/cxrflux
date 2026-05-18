import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)
        args = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LatentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LatentConditionalUNet(nn.Module):
    """Compact latent U-Net conditioned on image latents, treatment, delta, and timestep."""

    def __init__(
        self,
        latent_channels=4,
        conditioning_latents=1,
        base_channels=128,
        time_dim=256,
        conditioning_dim=256,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.conditioning_latents = conditioning_latents
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.treatment_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.delta_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

        in_channels = latent_channels * (1 + conditioning_latents)
        self.init = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = LatentResidualBlock(base_channels, base_channels, conditioning_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = LatentResidualBlock(base_channels * 2, base_channels * 2, conditioning_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = LatentResidualBlock(base_channels * 4, base_channels * 4, conditioning_dim)
        self.mid2 = LatentResidualBlock(base_channels * 4, base_channels * 4, conditioning_dim)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = LatentResidualBlock(base_channels * 4, base_channels * 2, conditioning_dim)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = LatentResidualBlock(base_channels * 2, base_channels, conditioning_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, noisy_latent, conditioning_latent, timesteps, treatment, delta):
        cond = (
            self.time_embedding(timesteps)
            + self.treatment_embedding(treatment.float())
            + self.delta_embedding(delta.float())
        )
        h0 = self.init(torch.cat([noisy_latent, conditioning_latent], dim=1))
        h1 = self.down1(h0, cond)
        h2 = self.down2(self.downsample1(h1), cond)
        h = self.downsample2(h2)
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)
        h = self.upsample2(h)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        return self.out(h)


class LatentMeanPredictor(nn.Module):
    """Deterministic latent follow-up predictor used as the DDPM mean anchor."""

    def __init__(
        self,
        latent_channels=4,
        base_channels=64,
        conditioning_dim=256,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.treatment_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.delta_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.init = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = LatentResidualBlock(base_channels, base_channels, conditioning_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.block2 = LatentResidualBlock(base_channels * 2, base_channels * 2, conditioning_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.block3 = LatentResidualBlock(base_channels * 2, base_channels, conditioning_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, baseline_latent, treatment, delta):
        cond = self.treatment_embedding(treatment.float()) + self.delta_embedding(delta.float())
        h1 = self.block1(self.init(baseline_latent), cond)
        h2 = self.block2(self.downsample(h1), cond)
        h = self.upsample(h2)
        h = self.block3(torch.cat([h, h1], dim=1), cond)
        return baseline_latent + self.out(h)


class LatentPotentialOutcomeMeanPredictor(nn.Module):
    """Mean predictor with explicit natural-progression and treatment-response fields.

    The factual mean is
        mu_a(z0, dt) = mu0(z0, dt) + a * tau(z0, dt)
    where mu0 is the no-treatment trajectory and tau is the latent treatment response.
    """

    supports_potential_outcomes = True

    def __init__(
        self,
        latent_channels=4,
        base_channels=64,
        conditioning_dim=256,
        treatment_effect_scale=1.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.treatment_effect_scale = treatment_effect_scale
        self.delta_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.init = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = LatentResidualBlock(base_channels, base_channels, conditioning_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.block2 = LatentResidualBlock(base_channels * 2, base_channels * 2, conditioning_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.block3 = LatentResidualBlock(base_channels * 2, base_channels, conditioning_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

        self.tau_delta_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.tau_init = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.tau_block1 = LatentResidualBlock(base_channels, base_channels, conditioning_dim)
        self.tau_downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.tau_block2 = LatentResidualBlock(base_channels * 2, base_channels * 2, conditioning_dim)
        self.tau_upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.tau_block3 = LatentResidualBlock(base_channels * 2, base_channels, conditioning_dim)
        self.tau_out = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        nn.init.zeros_(self.tau_out[-1].weight)
        nn.init.zeros_(self.tau_out[-1].bias)

    def _progression(self, baseline_latent, delta):
        cond = self.delta_embedding(delta.float())
        h1 = self.block1(self.init(baseline_latent), cond)
        h2 = self.block2(self.downsample(h1), cond)
        h = self.upsample(h2)
        h = self.block3(torch.cat([h, h1], dim=1), cond)
        return baseline_latent + self.out(h)

    def _treatment_response(self, baseline_latent, delta):
        cond = self.tau_delta_embedding(delta.float())
        h1 = self.tau_block1(self.tau_init(baseline_latent), cond)
        h2 = self.tau_block2(self.tau_downsample(h1), cond)
        h = self.tau_upsample(h2)
        h = self.tau_block3(torch.cat([h, h1], dim=1), cond)
        return self.treatment_effect_scale * self.tau_out(h)

    def forward(self, baseline_latent, treatment, delta, return_components=False):
        mu0 = self._progression(baseline_latent, delta)
        tau = self._treatment_response(baseline_latent, delta)
        treatment = treatment.float().view(-1, 1, 1, 1)
        mean = mu0 + treatment * tau
        if return_components:
            return mean, mu0, tau
        return mean


class LatentAnchoredPotentialOutcomeMeanPredictor(nn.Module):
    """Potential-outcome mean head anchored by a standard mean predictor.

    The no-treatment trajectory is represented by the same architecture used by
    the sharp V2 mean-residual model:

        mu_0(z0, dt) = standard_mean(z0, a=0, dt)

    and the treatment response is learned as an additive latent field:

        mu_a(z0, dt) = mu_0(z0, dt) + a * tau(z0, dt).
    """

    supports_potential_outcomes = True

    def __init__(
        self,
        latent_channels=4,
        base_channels=64,
        conditioning_dim=256,
        treatment_effect_scale=1.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.treatment_effect_scale = treatment_effect_scale
        self.progression = LatentMeanPredictor(
            latent_channels=latent_channels,
            base_channels=base_channels,
            conditioning_dim=conditioning_dim,
        )

        self.tau_delta_embedding = nn.Sequential(
            nn.Linear(1, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.tau_init = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.tau_block1 = LatentResidualBlock(base_channels, base_channels, conditioning_dim)
        self.tau_downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.tau_block2 = LatentResidualBlock(base_channels * 2, base_channels * 2, conditioning_dim)
        self.tau_upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.tau_block3 = LatentResidualBlock(base_channels * 2, base_channels, conditioning_dim)
        self.tau_out = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.tau_out[-1].weight)
        nn.init.zeros_(self.tau_out[-1].bias)

    def _progression(self, baseline_latent, treatment, delta):
        no_treatment = torch.zeros_like(treatment.float())
        return self.progression(baseline_latent, no_treatment, delta)

    def _treatment_response(self, baseline_latent, delta):
        cond = self.tau_delta_embedding(delta.float())
        h1 = self.tau_block1(self.tau_init(baseline_latent), cond)
        h2 = self.tau_block2(self.tau_downsample(h1), cond)
        h = self.tau_upsample(h2)
        h = self.tau_block3(torch.cat([h, h1], dim=1), cond)
        return self.treatment_effect_scale * self.tau_out(h)

    def forward(self, baseline_latent, treatment, delta, return_components=False):
        mu0 = self._progression(baseline_latent, treatment, delta)
        tau = self._treatment_response(baseline_latent, delta)
        treatment = treatment.float().view(-1, 1, 1, 1)
        mean = mu0 + treatment * tau
        if return_components:
            return mean, mu0, tau
        return mean


def cosine_beta_schedule(num_timesteps, s=0.008):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(num_timesteps):
    return torch.linspace(1e-4, 0.02, num_timesteps)


def extract(values, timesteps, shape):
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class LatentDDPM(nn.Module):
    def __init__(self, num_timesteps=1000, beta_schedule="cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def get_v_target(self, x_start, timesteps, noise):
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * noise - sqrt_one_minus * x_start

    def predict_x0_and_eps(self, x_t, timesteps, v_prediction):
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        x0 = sqrt_alpha * x_t - sqrt_one_minus * v_prediction
        eps = sqrt_one_minus * x_t + sqrt_alpha * v_prediction
        return x0, eps

    def p_losses(self, model, target_latent, timesteps, conditioning_latent, treatment, delta, weights=None):
        noise = torch.randn_like(target_latent)
        noisy = self.q_sample(target_latent, timesteps, noise)
        v_target = self.get_v_target(target_latent, timesteps, noise)
        v_prediction = model(noisy, conditioning_latent, timesteps, treatment, delta)
        loss = (v_prediction - v_target).pow(2).mean(dim=(1, 2, 3))
        if weights is not None:
            weights = weights.view(-1).float()
            loss = loss * weights
        return loss.mean()

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        conditioning_latent,
        treatment,
        delta,
        start_timestep=250,
        steps=50,
        noise=None,
        start_latent=None,
        latent_clip=8.0,
    ):
        model.eval()
        device = conditioning_latent.device
        batch_size = conditioning_latent.shape[0]
        if start_latent is None:
            start_latent = conditioning_latent[:, : model.latent_channels]
        start_timestep = min(max(int(start_timestep), 1), self.num_timesteps - 1)
        if noise is None:
            noise = torch.randn_like(start_latent)
        start_t = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
        x = self.q_sample(start_latent, start_t, noise=noise)

        timesteps = torch.linspace(start_timestep, 0, steps + 1, device=device).long().unique(sorted=True)
        timesteps = torch.flip(timesteps, dims=[0])
        if timesteps[0].item() != start_timestep:
            timesteps = torch.cat([torch.tensor([start_timestep], device=device), timesteps])

        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            t = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), int(next_timestep.item()), device=device, dtype=torch.long)
            v_prediction = model(x, conditioning_latent, t, treatment, delta)
            x0, eps = self.predict_x0_and_eps(x, t, v_prediction)
            if latent_clip is not None and latent_clip > 0:
                x0 = x0.clamp(-latent_clip, latent_clip)
            alpha_next = extract(self.sqrt_alphas_cumprod, t_next, x.shape)
            one_minus_next = extract(self.sqrt_one_minus_alphas_cumprod, t_next, x.shape)
            x = alpha_next * x0 + one_minus_next * eps
        return x.clamp(-latent_clip, latent_clip) if latent_clip is not None else x
