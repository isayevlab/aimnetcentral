"""Compute deterministic full-model energies/forces on CUDA and dump to JSON.

Version-agnostic: the orchestrator runs this once inside each per-torch-version
venv. CUDA is required (the Warp kernels execute only on GPU); without it the
entrypoint exits with a clear message so the orchestrator can record the leg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version


def _ver(dist: str):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def resolved_versions() -> dict:
    return {
        "torch": _ver("torch"),
        "triton": _ver("triton"),
        "warp-lang": _ver("warp-lang"),
        "nvalchemi-toolkit-ops": _ver("nvalchemi-toolkit-ops"),
    }


def _pin_determinism() -> None:
    import torch

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def _repo_data_dir() -> str:
    # tests/data lives two levels up from this file: <repo>/aimnet/validation/ -> <repo>/tests/data
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "tests", "data"))


def build_systems() -> dict:
    """Return ``{name: data}`` for the fixed validation set.

    water/methane are inline; caffeine and the spiro crystal load from
    tests/data via ASE (matching tests/conftest.py fixtures).
    """
    import ase.io

    data = {
        "water": {
            "coord": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            "numbers": [8, 1, 1],
            "charge": 0.0,
        },
        "methane": {
            "coord": [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            "numbers": [6, 1, 1, 1, 1],
            "charge": 0.0,
        },
    }

    ddir = _repo_data_dir()
    caffeine = ase.io.read(os.path.join(ddir, "caffeine.xyz"))
    data["caffeine"] = {
        "coord": caffeine.get_positions().tolist(),
        "numbers": caffeine.get_atomic_numbers().tolist(),
        "charge": 0.0,
    }
    crystal = ase.io.read(os.path.join(ddir, "2000054.cif"))
    data["spiro_pbc"] = {
        "coord": crystal.get_positions().tolist(),
        "numbers": crystal.get_atomic_numbers().tolist(),
        "charge": 0.0,
        "cell": crystal.get_cell().array.tolist(),
        "pbc": [True, True, True],
    }
    return data


def compute_observables() -> dict:
    """Build the calculator on CUDA and return {name: {energy, forces}}."""
    from aimnet.calculators import AIMNet2Calculator

    _pin_determinism()
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cuda")
    out = {}
    for name, data in build_systems().items():
        res = calc(data, forces=True)
        energy = float(res["energy"].detach().double().cpu().reshape(-1)[0].item())
        # Single-structure output is (N, 3) with no batch dim — do not squeeze.
        forces = res["forces"].detach().double().cpu().tolist()
        out[name] = {"energy": energy, "forces": forces}
    return out


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="path to write the observables JSON")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("gpu_observables requires CUDA; none available on this host.")

    doc = {"versions": resolved_versions(), "systems": compute_observables()}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {args.out} (torch {doc['versions']['torch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
