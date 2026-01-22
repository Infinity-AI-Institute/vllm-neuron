# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

# Datasets live under FSX_TEAM_SHARED_RO/datasets/trn2_datasets
_DATASETS_SUBDIR = "datasets/trn2_datasets"

# Simple map from friendly name -> filename on FSX
_DATASETS_MAPPING = {
    "MATH": "math-math.json",
    "HUMANEVAL": "HumanEval_python_sample_1000.json",
    "LONGBENCH_COT": "longbench_100k_120k_cot.json",
    "DYNAMIC_SONNET_256_1024_1024": "dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.json",
    "DYNAMIC_SONNET_512_2048_1024": "dynamic_sonnet_llama_3_prefix_512_max_2048_1024_sampled.json",
    "DYNAMIC_SONNET_1024_4096_1024": "dynamic_sonnet_llama_3_prefix_1024_max_4096_1024_sampled.json",
    "DYNAMIC_SONNET_2048_8192_1024": "dynamic_sonnet_llama_3_prefix_2048_max_8192_1024_sampled.json",
    "32K_65K": "32k_65k.json",
    "65K_131K": "65k_131k.json",
}


def _env(name: str) -> str:
    return os.environ.get(name, "") or ""


def _roots() -> Tuple[Path, Optional[Path]]:
    fsx_root = _env("FSX_TEAM_SHARED_RO")
    if not fsx_root:
        raise FileNotFoundError("FSX_TEAM_SHARED_RO is not set (required for datasets)")
    fsx_dir = Path(fsx_root) / _DATASETS_SUBDIR
    ssd = _env("SSD_RW")
    ssd_dir = (Path(ssd) / _DATASETS_SUBDIR) if ssd else None
    if ssd_dir:
        ssd_dir.mkdir(parents=True, exist_ok=True)
    return fsx_dir, ssd_dir


def list_available() -> dict:
    """Return a dict of friendly name -> absolute FSX path (no SSD copy)."""
    fsx_dir, _ = _roots()
    out = {}
    for name, fname in _DATASETS_MAPPING.items():
        p = fsx_dir / fname
        if p.exists():
            out[name] = str(p)
    return out


def resolve_dataset(spec: Optional[str]) -> Optional[str]:
    """
    Accepts:
      - None / ""                    -> returns None (no dataset)
      - "MATH", "HumanEval", ...     -> friendly names (case-insensitive)
      - "math-math.json"             -> filename in FSX datasets dir
      - "/abs/path/to/file.json"     -> returned as-is (no copy)
    Returns an SSD-mirrored absolute path if possible; else FSX path; else None.
    """
    if not spec:
        return None

    s = spec.strip()
    if not s:
        return None

    # Absolute path: leave it alone
    if os.path.isabs(s):
        return s

    # Friendly name? normalize to upper & strip non-alnum/underscore
    key = s.upper()
    key = key.replace(" ", "_").replace("-", "_").replace(".", "")
    fname = _DATASETS_MAPPING.get(key)

    fsx_dir, ssd_dir = _roots()

    # If not a known friendly name, treat as a filename under FSX datasets dir
    if not fname:
        # if it already looks like a json filename, use directly
        if s.lower().endswith(".json"):
            candidate = fsx_dir / s
        else:
            # Unknown token; return None (caller can treat as no dataset)
            return None
    else:
        candidate = fsx_dir / fname

    if not candidate.exists():
        # not available -> no dataset
        return None

    # Mirror to SSD if we can
    if ssd_dir:
        dst = ssd_dir / candidate.name
        if not dst.exists() or candidate.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(candidate, dst)
        return str(dst)

    return str(candidate)
