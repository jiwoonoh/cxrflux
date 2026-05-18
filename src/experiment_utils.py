import random

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize_image(image_tensor):
    return image_tensor.detach().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_fixed_noise(batch_size, image_size, device, seed):
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return torch.randn(
        batch_size,
        1,
        image_size,
        image_size,
        generator=generator,
        device=device,
    )


def apply_conditioning_mode(treatment, conditioning_mode):
    if conditioning_mode == "full":
        return treatment
    if conditioning_mode == "no_treatment":
        return torch.zeros_like(treatment)
    raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")
