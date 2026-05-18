# CXRFlux

This folder contains the final code, model weights, and results for CXRFlux,

## Contents

- `src/`: model, training, evaluation, and cohort-building code
- `scripts/`: runnable training and evaluation scripts
- `configs/`: study configuration files
- `results/`: final V2 panels and metrics

## Main Commands

Evaluate the selected V2 model:

```bash
bash scripts/run_latent_diffusion_eval.sh
```

Run V2 diagnostics:

```bash
bash scripts/run_latent_diffusion_diagnostics.sh
```

Retrain the V2 mean-residual model:

```bash
bash scripts/run_target_trial_latent_diffusion_finetune.sh
```
The V2 mean-residual latent DDPM is the selected model. V1 is the direct diffusion baseline. V3 and V4 were explored but not used as the final model because their visual outputs were blurrier despite stronger scalar metrics.
