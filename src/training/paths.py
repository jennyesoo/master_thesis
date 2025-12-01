import os
import glob
from dataclasses import dataclass

@dataclass
class Paths:
    log_dir: str
    checkpoint_dir: str
    sample_dir: str

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def build_log_dir(base_log_dir: str, dataset: str, is_can: bool, lr: float,
                  imsize: int, batch_size: int) -> str:
    root = os.path.join(
        base_log_dir,
        f"dataset={dataset},isCan={is_can},lr={lr},imsize={imsize},batch_size={batch_size}"
    )
    pattern = os.path.join(root, "[0-9][0-9][0-9]")
    existed = glob.glob(pattern)
    if not existed:
        return os.path.join(root, "000")
    nums = sorted(int(p[-3:]) for p in existed if p[-3:].isdigit())
    next_id = (nums[-1] + 1) if nums else 0
    return os.path.join(root, f"{next_id:03d}")

def resolve_paths(base_log_dir: str,
                  dataset: str,
                  is_can: bool,
                  lr: float,
                  imsize: int,
                  batch_size: int,
                  checkpoint_dir: str | None,
                  sample_dir: str | None) -> Paths:
    log_dir = build_log_dir(base_log_dir, dataset, is_can, lr, imsize, batch_size)
    ckpt_dir = checkpoint_dir or os.path.join(log_dir, "checkpoint")
    samp_dir = sample_dir or os.path.join(log_dir, "samples")
    ensure_dirs(log_dir, ckpt_dir, samp_dir)
    return Paths(log_dir=log_dir, checkpoint_dir=ckpt_dir, sample_dir=samp_dir)
