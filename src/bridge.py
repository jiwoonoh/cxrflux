import math

import torch


class BridgeProcess:
    """
    Deterministic paired-image bridge for baseline-to-follow-up CXR prediction.

    The bridge state is an interpolation between the follow-up image and the
    baseline image:

        z_t = (1 - m_t) * y + m_t * x_0 + sigma_t * eps

    where t=0 is near the target follow-up and t=max is the clean baseline.
    This matches the inference problem better than pure-noise DDPM sampling.
    """

    def __init__(
        self,
        num_steps=100,
        noise_scale=0.06,
        endpoint_probability=0.25,
        residual_scale=1.0,
        residual_mask_floor=1.0,
        residual_edge_margin=0.06,
        residual_lower_taper_start=0.70,
    ):
        if num_steps < 2:
            raise ValueError("BridgeProcess requires num_steps >= 2")
        self.num_steps = int(num_steps)
        self.noise_scale = float(noise_scale)
        self.endpoint_probability = float(endpoint_probability)
        self.residual_scale = float(residual_scale)
        self.residual_mask_floor = float(residual_mask_floor)
        self.residual_edge_margin = float(residual_edge_margin)
        self.residual_lower_taper_start = float(residual_lower_taper_start)

    @property
    def max_timestep(self):
        return self.num_steps - 1

    def sample_timesteps(self, batch_size, device):
        timesteps = torch.randint(0, self.num_steps, (batch_size,), device=device)
        if self.endpoint_probability > 0.0:
            endpoint_mask = torch.rand(batch_size, device=device) < self.endpoint_probability
            endpoint = torch.full_like(timesteps, self.max_timestep)
            timesteps = torch.where(endpoint_mask, endpoint, timesteps)
        return timesteps

    def mixing_weight(self, timesteps):
        return timesteps.float() / float(self.max_timestep)

    def noise_std(self, timesteps):
        mix = self.mixing_weight(timesteps)
        return self.noise_scale * torch.sin(math.pi * mix)

    def q_sample(self, baseline, target, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(target)
        mix = self.mixing_weight(timesteps).view(-1, 1, 1, 1)
        sigma = self.noise_std(timesteps).view(-1, 1, 1, 1)
        state = (1.0 - mix) * target + mix * baseline + sigma * noise
        return state.clamp(-1.0, 1.0)

    def residual_gate(self, baseline):
        if self.residual_mask_floor >= 0.999:
            return torch.ones_like(baseline)

        height, width = baseline.shape[-2:]
        device = baseline.device
        dtype = baseline.dtype
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(
            1,
            1,
            height,
            1,
        )
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(
            1,
            1,
            1,
            width,
        )

        edge_margin = max(self.residual_edge_margin, 1e-4)
        left = (x / edge_margin).clamp(0.0, 1.0)
        right = ((1.0 - x) / edge_margin).clamp(0.0, 1.0)
        top = (y / edge_margin).clamp(0.0, 1.0)
        edge_taper = torch.minimum(torch.minimum(left, right), top)

        lower_start = min(max(self.residual_lower_taper_start, 0.0), 0.999)
        lower_taper = ((1.0 - y) / max(1.0 - lower_start, 1e-4)).clamp(0.0, 1.0)
        spatial_gate = edge_taper * lower_taper
        return self.residual_mask_floor + (1.0 - self.residual_mask_floor) * spatial_gate

    def predict_followup(self, model, bridge_state, timesteps, baseline, treatment, delta):
        raw_residual = model(bridge_state, timesteps, baseline, treatment, delta)
        residual = self.residual_scale * torch.tanh(raw_residual)
        residual = residual * self.residual_gate(baseline)
        return (baseline + residual).clamp(-1.0, 1.0)

    @torch.no_grad()
    def one_step_predict(self, model, baseline, treatment, delta):
        timesteps = torch.full(
            (baseline.shape[0],),
            self.max_timestep,
            dtype=torch.long,
            device=baseline.device,
        )
        return self.predict_followup(model, baseline, timesteps, baseline, treatment, delta)

    @torch.no_grad()
    def iterative_predict(self, model, baseline, treatment, delta, inference_steps=None):
        if inference_steps is None or inference_steps >= self.num_steps:
            timestep_values = list(range(self.max_timestep, -1, -1))
        else:
            positions = torch.linspace(
                self.max_timestep,
                0,
                steps=max(2, int(inference_steps)),
                device=baseline.device,
            )
            timestep_values = [int(round(value.item())) for value in positions]
            timestep_values = sorted(set(timestep_values), reverse=True)
            if timestep_values[-1] != 0:
                timestep_values.append(0)

        state = baseline
        prediction = baseline
        for index, timestep_value in enumerate(timestep_values):
            timesteps = torch.full(
                (baseline.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=baseline.device,
            )
            prediction = self.predict_followup(
                model,
                state,
                timesteps,
                baseline,
                treatment,
                delta,
            )
            if index + 1 < len(timestep_values):
                next_timestep = torch.full(
                    (baseline.shape[0],),
                    timestep_values[index + 1],
                    dtype=torch.long,
                    device=baseline.device,
                )
                mix = self.mixing_weight(next_timestep).view(-1, 1, 1, 1)
                state = ((1.0 - mix) * prediction + mix * baseline).clamp(-1.0, 1.0)

        return prediction
