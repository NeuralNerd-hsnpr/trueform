"""Run metadata: seeds, hardware, memory, git commit, file hashes."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import resource
import subprocess
from pathlib import Path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware() -> dict:
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {
        "cpu": cpu,
        "cpu_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    return info


def peak_memory_mb(device: str) -> dict:
    """Peak process RSS, and peak allocated GPU memory when on CUDA."""
    import torch

    out = {"cpu_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    if device.startswith("cuda"):
        out["gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def git_commit(root: Path) -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=root) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path, patterns=("*.safetensors", "*.json")) -> str:
    """Hash of the weight and config files in a checkpoint directory."""
    h = hashlib.sha256()
    files = sorted({p for pat in patterns for p in Path(path).glob(pat)})
    for p in files:
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()
