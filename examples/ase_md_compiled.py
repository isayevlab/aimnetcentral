"""Example: Molecular dynamics with torch.compile and CUDA graphs.

This example demonstrates the speedup from using torch.compile with CUDA graphs
for molecular dynamics simulations. On a modern GPU, compile mode can provide
~5x speedup for small molecules (76s -> 15s for 10k MD steps on caffeine).

Usage:
    python ase_md_compiled.py                  # Normal mode
    python ase_md_compiled.py --compile        # Compile mode with CUDA graphs

Requirements:
    - CUDA GPU
    - ASE (pip install aimnet[ase])
"""

import argparse
import os
from time import perf_counter

import torch


def torch_show_device_info():
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


def run_md(atoms, steps=1000, timestep=0.5, temperature=300):
    """Run molecular dynamics simulation."""
    from ase import units
    from ase.md.langevin import Langevin

    # Setup Langevin dynamics
    dyn = Langevin(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        friction=0.01 / units.fs,
    )

    # Run dynamics
    t0 = perf_counter()
    dyn.run(steps)
    t1 = perf_counter()

    return t1 - t0, dyn.nsteps


def main():
    parser = argparse.ArgumentParser(description="AIMNet2 MD with torch.compile")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile with CUDA graphs")
    parser.add_argument("--model", type=str, default="aimnet2", help="Model name")
    parser.add_argument("--steps", type=int, default=1000, help="Number of MD steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps (for compilation)")
    args = parser.parse_args()

    # Check CUDA availability for compile mode
    if args.compile and not torch.cuda.is_available():
        print("Error: --compile requires CUDA")
        return

    torch_show_device_info()
    print()

    # Load molecule
    import ase.io

    from aimnet.calculators import AIMNet2ASE

    xyzfile = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "caffeine.xyz")
    atoms = ase.io.read(xyzfile)

    print(f"Molecule: {len(atoms)} atoms")
    print(f"Model: {args.model}")
    print(f"Compile mode: {args.compile}")
    print()

    # Create calculator
    calc = AIMNet2ASE(args.model, compile_mode=args.compile)
    atoms.calc = calc

    # Warmup (especially important for compile mode to build CUDA graphs)
    print(f"Running {args.warmup} warmup steps...")
    warmup_time, _ = run_md(atoms, steps=args.warmup)
    print(f"Warmup completed in {warmup_time:.2f}s")

    if args.compile:
        print("(First run includes torch.compile compilation time)")
    print()

    # Main MD run
    print(f"Running {args.steps} MD steps...")
    elapsed, nsteps = run_md(atoms, steps=args.steps)

    print("\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per step: {elapsed / nsteps * 1000:.2f}ms")
    print(f"  Steps per second: {nsteps / elapsed:.1f}")

    # Get final energy
    energy = atoms.get_potential_energy()
    print(f"  Final energy: {energy:.4f} eV")


if __name__ == "__main__":
    main()
