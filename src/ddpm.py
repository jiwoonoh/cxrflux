# ddpm.py

import math

import torch
import torch.nn.functional as F


def _linear_beta_schedule(num_timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def _cosine_beta_schedule(num_timesteps, s=0.008):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="cosine",
        device="cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.beta_schedule = beta_schedule

        if beta_schedule == "linear":
            betas = _linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _extract(values, t, shape):
        return values[t].view(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, y_0, t, noise=None):
        """
        Forward process: q(y_t | y_0).
        """
        if noise is None:
            noise = torch.randn_like(y_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * y_0 + sqrt_one_minus_alpha * noise, noise

    def predict_x0_from_noise(self, y_t, t, noise_pred):
        """
        Convert an epsilon prediction into an estimate of the clean follow-up image.
        """
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas_cumprod, t, y_t.shape)
        sqrt_recipm1_alpha = self._extract(self.sqrt_recipm1_alphas_cumprod, t, y_t.shape)
        return sqrt_recip_alpha * y_t - sqrt_recipm1_alpha * noise_pred

    @staticmethod
    def _threshold_x0(x0_pred, clip_denoised, dynamic_threshold_percentile):
        if dynamic_threshold_percentile is not None:
            if not 0.0 < dynamic_threshold_percentile <= 1.0:
                raise ValueError("dynamic_threshold_percentile must be in (0, 1]")
            flat = x0_pred.detach().abs().flatten(start_dim=1)
            scale = torch.quantile(
                flat,
                dynamic_threshold_percentile,
                dim=1,
            ).clamp_min(1.0)
            scale = scale.view(-1, *((1,) * (x0_pred.ndim - 1)))
            return x0_pred.clamp(-scale, scale) / scale
        if clip_denoised:
            return x0_pred.clamp(-1.0, 1.0)
        return x0_pred

    def p_losses(self, model, y_0, t, x_0, a, delta, reduction="mean"):
        """
        Compute the diffusion training loss.
        """
        noise = torch.randn_like(y_0)
        y_t, _ = self.q_sample(y_0, t, noise)
        noise_pred = model(y_t, t, x_0, a, delta)
        per_example_loss = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3))
        if reduction == "none":
            return per_example_loss
        if reduction == "mean":
            return per_example_loss.mean()
        raise ValueError(f"Unsupported loss reduction: {reduction}")

    @torch.no_grad()
    def p_sample(
        self,
        model,
        y_t,
        t,
        x_0,
        a,
        delta,
        stochastic=True,
        clip_denoised=False,
        dynamic_threshold_percentile=None,
    ):
        """
        One reverse-diffusion step.
        """
        noise_pred = model(y_t, t, x_0, a, delta)
        if clip_denoised or dynamic_threshold_percentile is not None:
            x0_pred = self.predict_x0_from_noise(y_t, t, noise_pred)
            x0_pred = self._threshold_x0(
                x0_pred,
                clip_denoised=clip_denoised,
                dynamic_threshold_percentile=dynamic_threshold_percentile,
            )
            mean = (
                self._extract(self.posterior_mean_coef1, t, y_t.shape) * x0_pred
                + self._extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
            )
        else:
            betas_t = self._extract(self.betas, t, y_t.shape)
            sqrt_one_minus_alpha_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                y_t.shape,
            )
            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, y_t.shape)
            mean = sqrt_recip_alpha_t * (
                y_t - betas_t * noise_pred / sqrt_one_minus_alpha_t
            )

        if stochastic:
            noise = torch.randn_like(y_t)
            posterior_var = self._extract(self.posterior_variance, t, y_t.shape)
            nonzero_mask = (t != 0).float().view(t.shape[0], *((1,) * (y_t.ndim - 1)))
            return mean + nonzero_mask * torch.sqrt(posterior_var) * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        model,
        x_0,
        a,
        delta,
        image_size=128,
        initial_noise=None,
        initial_image=None,
        start_timestep=None,
        stochastic=True,
        clip_denoised=False,
        dynamic_threshold_percentile=None,
    ):
        """
        Sample a follow-up image from pure noise, a fixed initial noise tensor, or
        a partially noised initial image.
        When ``stochastic`` is False, reverse diffusion follows the posterior
        mean path exactly, which removes per-step sampling variance.
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        if start_timestep is None:
            start_timestep = self.num_timesteps - 1
        start_timestep = int(start_timestep)
        if not 0 <= start_timestep < self.num_timesteps:
            raise ValueError(
                f"start_timestep must be in [0, {self.num_timesteps - 1}], "
                f"got {start_timestep}"
            )

        if initial_image is not None:
            initial_image = initial_image.to(device)
            if initial_noise is None:
                noise = torch.randn_like(initial_image)
            else:
                noise = initial_noise.to(device).clone()
                if noise.shape != initial_image.shape:
                    raise ValueError("initial_noise shape must match initial_image shape")
            t_batch = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
            y_t, _ = self.q_sample(initial_image, t_batch, noise=noise)
        elif initial_noise is None:
            y_t = torch.randn(batch_size, 1, image_size, image_size, device=device)
        else:
            y_t = initial_noise.to(device).clone()

        for timestep in reversed(range(start_timestep + 1)):
            t_batch = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            y_t = self.p_sample(
                model,
                y_t,
                t_batch,
                x_0,
                a,
                delta,
                stochastic=stochastic,
                clip_denoised=clip_denoised,
                dynamic_threshold_percentile=dynamic_threshold_percentile,
            )

        return y_t.clamp(-1.0, 1.0)


if __name__ == "__main__":
    from unet import ConditionalUNet

    device = "cpu"
    model = ConditionalUNet().to(device)
    ddpm = DDPM(num_timesteps=1000, beta_schedule="cosine", device=device)

    batch_size = 2
    y_0 = torch.randn(batch_size, 1, 128, 128).to(device)
    x_0 = torch.randn(batch_size, 1, 128, 128).to(device)
    a = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    delta = torch.rand(batch_size, 1).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    loss = ddpm.p_losses(model, y_0, t, x_0, a, delta)
    print(f"Loss: {loss.item():.4f}")

    ddpm_fast = DDPM(num_timesteps=10, beta_schedule="cosine", device=device)
    sample = ddpm_fast.sample(model, x_0, a, delta)
    print(f"Sample shape: {sample.shape}")
