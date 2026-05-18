from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SUPPORTED_MASK_SUFFIXES = (".pt", ".pth", ".npy", ".png", ".jpg", ".jpeg")


def mask_path_for_dicom(mask_root, dicom_id):
    """Return the first cached mask path found for a DICOM id."""
    root = Path(mask_root)
    dicom_id = str(dicom_id)
    for suffix in SUPPORTED_MASK_SUFFIXES:
        path = root / f"{dicom_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No cached lung mask found for {dicom_id} in {root}. "
        f"Expected one of: {', '.join(dicom_id + s for s in SUPPORTED_MASK_SUFFIXES)}"
    )


def _tensor_from_loaded_mask(obj):
    if isinstance(obj, dict):
        for key in ("lung_mask", "mask", "segmentation"):
            if key in obj:
                obj = obj[key]
                break
        else:
            tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
            if len(tensor_values) != 1:
                raise ValueError("Mask checkpoint dict must contain a single tensor or a lung_mask key")
            obj = tensor_values[0]

    if torch.is_tensor(obj):
        tensor = obj.detach().float().cpu()
    else:
        tensor = torch.as_tensor(obj, dtype=torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            if tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor[:1]
    else:
        raise ValueError(f"Expected 2D or 3D mask tensor, got shape {tuple(tensor.shape)}")

    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def load_lung_mask(mask_root, dicom_id, image_size=None):
    """Load a cached lung mask as a [1, H, W] float tensor in [0, 1]."""
    path = mask_path_for_dicom(mask_root, dicom_id)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        tensor = _tensor_from_loaded_mask(obj)
    elif suffix == ".npy":
        tensor = _tensor_from_loaded_mask(np.load(path))
    else:
        with Image.open(path) as image:
            tensor = _tensor_from_loaded_mask(np.array(image.convert("L"), dtype=np.float32))

    if image_size is not None and tuple(tensor.shape[-2:]) != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def load_lung_mask_batch(mask_root, dicom_ids, image_size=None, device=None, dtype=torch.float32):
    masks = [load_lung_mask(mask_root, dicom_id, image_size=image_size) for dicom_id in dicom_ids]
    batch = torch.stack(masks, dim=0).to(dtype=dtype)
    if device is not None:
        batch = batch.to(device)
    return batch
