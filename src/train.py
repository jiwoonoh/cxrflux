# train.py

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXRPairDataset, build_transform, get_dataloaders
from ddpm import DDPM
from experiment_utils import (
    apply_conditioning_mode,
    denormalize_image,
    make_fixed_noise,
    move_batch_to_device,
    set_seed,
)
from unet import ConditionalUNet


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()

        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

def create_model(config, device):
    model = ConditionalUNet(
        in_channels=1,
        base_channels=config["base_channels"],
    ).to(device)
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    return model


def build_fixed_batch(pairs_df, image_root, image_size, sample_count, weight_column=None):
    sample_count = min(sample_count, len(pairs_df))
    transform = build_transform(image_size)
    dataset = CXRPairDataset(
        pairs_df.iloc[:sample_count].copy(),
        image_root,
        transform=transform,
        weight_column=weight_column,
    )
    loader = DataLoader(dataset, batch_size=sample_count, shuffle=False, num_workers=0)
    return next(iter(loader))


def evaluate(model, ddpm, loader, device, use_amp, conditioning_mode):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            conditioned_treatment = apply_conditioning_mode(
                batch["a"], conditioning_mode
            )
            timesteps = torch.randint(
                0,
                ddpm.num_timesteps,
                (batch["x_0"].shape[0],),
                device=device,
            )
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = ddpm.p_losses(
                    model,
                    batch["y"],
                    timesteps,
                    batch["x_0"],
                    conditioned_treatment,
                    batch["delta"],
                )
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(path, epoch, loss, model, ema, optimizer, scaler, config):
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "config": config,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema.ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
    }
    torch.save(checkpoint, path)


def save_comparison_panel(
    model,
    ddpm,
    batch,
    device,
    image_size,
    initial_noise,
    output_path,
    conditioning_mode,
    deterministic_sampling=False,
):
    model.eval()
    device_batch = move_batch_to_device(batch, device)
    conditioned_treatment = apply_conditioning_mode(
        device_batch["a"], conditioning_mode
    )

    with torch.no_grad():
        generated = ddpm.sample(
            model,
            device_batch["x_0"],
            conditioned_treatment,
            device_batch["delta"],
            image_size=image_size,
            initial_noise=initial_noise,
            stochastic=not deterministic_sampling,
        )

    batch_size = generated.shape[0]
    fig, axes = plt.subplots(3, batch_size, figsize=(4 * batch_size, 9))
    if batch_size == 1:
        axes = axes.reshape(3, 1)

    generated_cpu = generated.cpu()
    for index in range(batch_size):
        baseline = denormalize_image(batch["x_0"][index])[0]
        ground_truth = denormalize_image(batch["y"][index])[0]
        prediction = denormalize_image(generated_cpu[index])[0]

        axes[0, index].imshow(baseline, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, index].set_title(
            f"Baseline (a={int(batch['a'][index].item())}, Δ={batch['hours_diff'][index].item():.0f}h)"
        )
        axes[1, index].imshow(ground_truth, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, index].set_title("Ground Truth")
        axes[2, index].imshow(prediction, cmap="gray", vmin=0.0, vmax=1.0)
        title = "Generated" if conditioning_mode == "full" else "Generated (A masked)"
        axes[2, index].set_title(title)

        for row in range(3):
            axes[row, index].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_treatment_swap_panel(
    model,
    ddpm,
    batch,
    device,
    image_size,
    initial_noise,
    output_path,
    conditioning_mode,
    deterministic_sampling=True,
):
    model.eval()
    device_batch = move_batch_to_device(batch, device)
    factual_request = device_batch["a"]
    swapped_request = 1.0 - device_batch["a"]
    factual_treatment = apply_conditioning_mode(factual_request, conditioning_mode)
    swapped_treatment = apply_conditioning_mode(swapped_request, conditioning_mode)

    with torch.no_grad():
        factual = ddpm.sample(
            model,
            device_batch["x_0"],
            factual_treatment,
            device_batch["delta"],
            image_size=image_size,
            initial_noise=initial_noise,
            stochastic=not deterministic_sampling,
        )
        counterfactual = ddpm.sample(
            model,
            device_batch["x_0"],
            swapped_treatment,
            device_batch["delta"],
            image_size=image_size,
            initial_noise=initial_noise,
            stochastic=not deterministic_sampling,
        )

    batch_size = factual.shape[0]
    fig, axes = plt.subplots(4, batch_size, figsize=(4 * batch_size, 12))
    if batch_size == 1:
        axes = axes.reshape(4, 1)

    factual_cpu = factual.cpu()
    counterfactual_cpu = counterfactual.cpu()
    for index in range(batch_size):
        baseline = denormalize_image(batch["x_0"][index])[0]
        ground_truth = denormalize_image(batch["y"][index])[0]
        factual_image = denormalize_image(factual_cpu[index])[0]
        counterfactual_image = denormalize_image(counterfactual_cpu[index])[0]

        actual_treatment = int(factual_request[index].item())
        flipped_treatment = 1 - actual_treatment
        hours = batch["hours_diff"][index].item()

        axes[0, index].imshow(baseline, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, index].set_title(f"Baseline (Δ={hours:.0f}h)")
        axes[1, index].imshow(ground_truth, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, index].set_title("Ground Truth")
        axes[2, index].imshow(factual_image, cmap="gray", vmin=0.0, vmax=1.0)
        if conditioning_mode == "full":
            factual_title = f"Generated factual (a={actual_treatment})"
            swapped_title = f"Generated swapped (a={flipped_treatment})"
        else:
            factual_title = f"Generated factual (A masked; requested a={actual_treatment})"
            swapped_title = f"Generated swapped (A masked; requested a={flipped_treatment})"
        axes[2, index].set_title(factual_title)
        axes[3, index].imshow(counterfactual_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[3, index].set_title(swapped_title)

        for row in range(4):
            axes[row, index].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_checkpoint_for_eval(checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(config, device)
    state_dict = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def train(config):
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["use_amp"] and device.type == "cuda"
    print(f"Using device: {device}")
    print(f"Conditioning mode: {config['conditioning_mode']}")
    if config["loss_weight_column"]:
        print(f"Loss weight column: {config['loss_weight_column']}")

    sample_dir = Path(config["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    Path(config["save_path"]).parent.mkdir(parents=True, exist_ok=True)

    loaders, split_frames = get_dataloaders(
        csv_path=config["csv_path"],
        image_root=config["image_root"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        split_dir=config["split_dir"],
        split_fractions=(
            config["train_fraction"],
            config["val_fraction"],
            config["test_fraction"],
        ),
        weight_column=config["loss_weight_column"],
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    fixed_val_batch = build_fixed_batch(
        split_frames["val"],
        config["image_root"],
        config["image_size"],
        config["sample_count"],
        weight_column=config["loss_weight_column"],
    )
    fixed_test_batch = build_fixed_batch(
        split_frames["test"],
        config["image_root"],
        config["image_size"],
        config["sample_count"],
        weight_column=config["loss_weight_column"],
    )

    val_noise = make_fixed_noise(
        fixed_val_batch["x_0"].shape[0],
        config["image_size"],
        device,
        config["sample_seed"],
    )
    test_noise = make_fixed_noise(
        fixed_test_batch["x_0"].shape[0],
        config["image_size"],
        device,
        config["sample_seed"] + 1,
    )

    model = create_model(config, device)
    ema = ExponentialMovingAverage(model, decay=config["ema_decay"])
    ddpm = DDPM(
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(config["epochs"]):
        model.train()
        running_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            conditioned_treatment = apply_conditioning_mode(
                batch["a"], config["conditioning_mode"]
            )
            timesteps = torch.randint(
                0,
                config["num_timesteps"],
                (batch["x_0"].shape[0],),
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if config["loss_weight_column"]:
                    per_example_loss = ddpm.p_losses(
                        model,
                        batch["y"],
                        timesteps,
                        batch["x_0"],
                        conditioned_treatment,
                        batch["delta"],
                        reduction="none",
                    )
                    sample_weight = batch["sample_weight"].view(-1)
                    loss = (per_example_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
                else:
                    loss = ddpm.p_losses(
                        model,
                        batch["y"],
                        timesteps,
                        batch["x_0"],
                        conditioned_treatment,
                        batch["delta"],
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_train_loss / max(len(train_loader), 1)
        val_loss = evaluate(
            ema.ema_model,
            ddpm,
            val_loader,
            device,
            use_amp,
            config["conditioning_mode"],
        )

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                config["save_path"],
                epoch,
                best_val_loss,
                model,
                ema,
                optimizer,
                scaler,
                config,
            )
            print(f"Saved best model (loss={best_val_loss:.4f})")

            save_comparison_panel(
                ema.ema_model,
                ddpm,
                fixed_val_batch,
                device,
                config["image_size"],
                val_noise,
                sample_dir / "best_val_samples.png",
                config["conditioning_mode"],
            )
            save_treatment_swap_panel(
                ema.ema_model,
                ddpm,
                fixed_val_batch,
                device,
                config["image_size"],
                val_noise,
                sample_dir / "best_val_treatment_swap.png",
                config["conditioning_mode"],
            )

        if config["save_every"] > 0 and (epoch + 1) % config["save_every"] == 0:
            checkpoint_path = Path(config["save_path"]).with_name(
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                epoch,
                val_loss,
                model,
                ema,
                optimizer,
                scaler,
                config,
            )
            print(f"Saved {checkpoint_path.name}")

        if (epoch + 1) % config["sample_every"] == 0:
            save_comparison_panel(
                ema.ema_model,
                ddpm,
                fixed_val_batch,
                device,
                config["image_size"],
                val_noise,
                sample_dir / f"epoch_{epoch + 1:03d}.png",
                config["conditioning_mode"],
            )

    print(f"Best validation checkpoint: epoch {best_epoch + 1}, loss={best_val_loss:.4f}")

    best_model, checkpoint = load_checkpoint_for_eval(config["save_path"], config, device)
    test_loss = evaluate(
        best_model,
        ddpm,
        test_loader,
        device,
        use_amp,
        config["conditioning_mode"],
    )
    print(f"Best checkpoint test_loss={test_loss:.4f}")

    save_comparison_panel(
        best_model,
        ddpm,
        fixed_test_batch,
        device,
        config["image_size"],
        test_noise,
        sample_dir / "best_test_samples.png",
        config["conditioning_mode"],
    )
    save_treatment_swap_panel(
        best_model,
        ddpm,
        fixed_test_batch,
        device,
        config["image_size"],
        test_noise,
        sample_dir / "best_test_treatment_swap.png",
        config["conditioning_mode"],
    )

    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional CXR DDPM")
    parser.add_argument("--csv-path", default="cxr_pairs_frontal.csv")
    parser.add_argument("--image-root", default="./mimic-cxr-jpg/")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--save-path", default="best_model.pt")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--conditioning-mode", choices=["full", "no_treatment"], default="full")
    parser.add_argument("--loss-weight-column", default=None)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    if args.split_dir is None:
        csv_stem = Path(args.csv_path).stem
        args.split_dir = str(Path("splits") / f"{csv_stem}_seed{args.seed}")

    return {
        "csv_path": args.csv_path,
        "image_root": args.image_root,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "num_timesteps": args.num_timesteps,
        "beta_schedule": args.beta_schedule,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "save_path": args.save_path,
        "save_every": args.save_every,
        "sample_every": args.sample_every,
        "sample_count": args.sample_count,
        "sample_seed": args.sample_seed,
        "sample_dir": args.sample_dir,
        "ema_decay": args.ema_decay,
        "split_dir": args.split_dir,
        "conditioning_mode": args.conditioning_mode,
        "loss_weight_column": args.loss_weight_column,
        "use_amp": not args.disable_amp,
    }


if __name__ == "__main__":
    train(parse_args())
