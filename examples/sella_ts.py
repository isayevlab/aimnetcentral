"""Sella transition-state search using AIMNet2's analytic Hessian.

Demonstrates the recommended configuration for using Sella with AIMNet2:
- Minimum-mode following on internal coordinates (order=1, internal=True).
- Analytic Hessian callback via AIMNet2ASE.get_hessian — replaces each
  Davidson refresh with one analytic Hessian call (O(3N) backward passes
  per refresh through the AIMNet2 energy graph).

Reference: Yuan et al., Nature Communications 2024
(https://www.nature.com/articles/s41467-024-52481-5) showed that providing
analytic ML Hessians to Sella reduces step count by 2-3x.

Requires: pip install "aimnet[sella]"
"""

from time import perf_counter

import ase.io
from sella import Sella

from aimnet.calculators import AIMNet2ASE


def main(xyz_path: str, fmax: float = 0.01, max_steps: int = 200) -> None:
    atoms = ase.io.read(xyz_path)
    atoms.calc = AIMNet2ASE("aimnet2")

    dyn = Sella(
        atoms,
        order=1,
        internal=True,
        hessian_function=atoms.calc.get_hessian,
    )

    print(f"Sella TS search for {len(atoms)} atoms; fmax={fmax} eV/A.")
    t0 = perf_counter()
    dyn.run(fmax=fmax, steps=max_steps)
    t1 = perf_counter()

    nsteps = dyn.nsteps
    print(f"Converged in {nsteps} steps ({t1 - t0:.1f} s, {(t1 - t0) / max(nsteps, 1):.3f} s/step).")
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")

    ase.io.write("ts_optimized.xyz", atoms)
    print("Optimized geometry written to ts_optimized.xyz.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python sella_ts.py <ts_guess.xyz>")
        sys.exit(1)
    main(sys.argv[1])
