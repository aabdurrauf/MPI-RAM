
import argparse
import glob
import os
import sys

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ram.mpi.utils.mpiSuperResPhysics import MPISuperResPhysics
from utils.loadData import *
from ram.models.ram import RAM


def _pick_latest_checkpoint(ckpt_dir: str) -> str:
    ckpt_glob = os.path.join(ckpt_dir, "*.tar")
    candidates = glob.glob(ckpt_glob)
    if not candidates:
        raise FileNotFoundError(f"No .tar checkpoints found in: {ckpt_dir}")
    return max(candidates, key=os.path.getmtime)


def _load_checkpoint(model: RAM, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _to_magnitude_2ch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        real = x[0]
        imag = x[1]
    else:
        real = x[:, 0]
        imag = x[:, 1]
    return torch.sqrt(real ** 2 + imag ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RAM checkpoint.")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to a .tar checkpoint. If omitted, uses the latest under project root.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=PROJECT_ROOT,
        help="Directory to scan for .tar checkpoints when --ckpt is not set.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto (default).")
    args = parser.parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = args.ckpt or _pick_latest_checkpoint(args.ckpt_dir)


    # Load evaluation inputs from your dataset utilities.
    file_dir = "ram/mpi/data/train/"
    scale_factor = 2
    bs = 64
    evalDS, copymax, copymin = get_eval_data(file_dir, scale_factor, bs, 32, 32, 5)

    eval_model = RAM(device=str(device)).to(device)
    _load_checkpoint(eval_model, ckpt_path, device)
    eval_model.eval()

    y = evalDS.LR.to(device)
    x_gt = evalDS.HR.to(device)

    evalDS_bs = MpiDataset(
        x_gt.clone(),
        y.clone(),
        maxhr=copymax,
        minhr=copymin,
        batch_size=bs,
        up=0.7,
        down=0.15
    )

    physics = MPISuperResPhysics(
        dataset=evalDS_bs,
        scale_factor=scale_factor
    )

    with torch.no_grad():
        x_eval = eval_model(y, physics=physics)

    num_show = 3
    show_idx = torch.linspace(0, y.shape[0] - 1, num_show).long()

    fig, axes = plt.subplots(3, num_show, figsize=(4 * num_show, 10))
    for col, idx in enumerate(show_idx):
        lr_mag = _to_magnitude_2ch(y[idx]).cpu().numpy()
        gt_mag = _to_magnitude_2ch(x_gt[idx]).cpu().numpy()
        pred_mag = _to_magnitude_2ch(x_eval[idx]).cpu().numpy()

        axes[0, col].imshow(lr_mag, cmap="gray")
        axes[0, col].set_title(f"LR input {idx.item()}")
        axes[0, col].axis("off")

        axes[1, col].imshow(gt_mag, cmap="gray")
        axes[1, col].set_title("HR ground truth")
        axes[1, col].axis("off")

        axes[2, col].imshow(pred_mag, cmap="gray")
        axes[2, col].set_title("Prediction")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Loaded checkpoint: {ckpt_path}")
    print("Reloaded model inference done.")
    print(f"Max |x_gt - x_eval| = {(x_gt - x_eval).abs().max().item():.6e}")


if __name__ == "__main__":
    main()