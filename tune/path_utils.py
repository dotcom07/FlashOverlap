import re
from pathlib import Path
from typing import List, Tuple

import torch


def normalize_gpu_name(name: str) -> str:
    tag = name.strip().lower()
    if tag.startswith("nvidia "):
        tag = tag[len("nvidia "):]
    tag = re.sub(r"[^a-z0-9]+", "_", tag).strip("_")
    return tag or "unknown_gpu"


def current_gpu_model_tag(device: int = None) -> str:
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return normalize_gpu_name(props.name)


def legacy_gpu_tag(device: int = None) -> str:
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.name[7:11].lower()


def config_paths(
    m: int, n: int, k: int, config_dir: str = "../configs", device: int = None
) -> Tuple[Path, List[Path]]:
    root = Path(config_dir)
    model_tag = current_gpu_model_tag(device)
    preferred = root / f"m{m}n{n}k{k}_{model_tag}.json"

    candidates = [preferred]
    legacy = root / f"m{m}n{n}k{k}_{legacy_gpu_tag(device)}.json"
    if legacy != preferred:
        candidates.append(legacy)
    return preferred, candidates


def resolve_existing_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_config_load_path(
    m: int, n: int, k: int, config_dir: str = "../configs", device: int = None
) -> Path:
    preferred, candidates = config_paths(m, n, k, config_dir, device)
    return resolve_existing_path(candidates) or preferred


def resolve_config_save_path(
    m: int, n: int, k: int, config_dir: str = "../configs", device: int = None
) -> Path:
    preferred, _ = config_paths(m, n, k, config_dir, device)
    return preferred


def profile_csv_candidates(
    m: int, n: int, k: int, profile_dir: str, device: int = None
) -> Tuple[Path, List[Path]]:
    root = Path(profile_dir).expanduser()
    model_tag = current_gpu_model_tag(device)
    legacy_tag = legacy_gpu_tag(device)

    names = [
        f"m{m}n{n}k{k}_{model_tag}.gemm.csv",
        f"m{m}n{n}k{k}_{model_tag}.csv",
        f"m{m}n{n}k{k}_{legacy_tag}.gemm.csv",
        f"m{m}n{n}k{k}_{legacy_tag}.csv",
        f"m{m}n{n}k{k}.gemm.csv",
        f"m{m}n{n}k{k}.csv",
    ]
    dedup_names = list(dict.fromkeys(names))
    candidates = [root / name for name in dedup_names]
    return candidates[0], candidates
