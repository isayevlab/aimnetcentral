"""Reproduce Table 1 from Wu & Van Voorhis (JPCA 2006, 110, 9212)
using AIMNet2 + CQEq for the Q-TTF-Q radical anion.

The tetrathiafulvalene-diquinone (TTF-DQ or Q-TTF-Q) radical anion is a
symmetric mixed-valence compound.  Unconstrained DFT (BLYP) incorrectly
predicts a delocalized (Robin-Day class III) ground state, while CDFT
gives the experimentally correct charge-localized (class II) structure.

Here we reproduce the same analysis with AIMNet2 + CQEq:

1. Optimize the adiabatic ground state (unconstrained, charge = -1).
2. Optimize diabatic state 1: Q^-_left-TTF-Q_right  (left=-1, right=0).
3. Optimize diabatic state 2: Q_left-TTF-Q^-_right  (left=0, right=-1).
4. Extract selected bond lengths -> Table 1.
5. Compute Marcus electron-transfer parameters: Delta-G, lambda_i, H_ab.

Reference
---------
Q. Wu & T. Van Voorhis, "Direct Calculation of Electron Transfer
Parameters through Constrained Density Functional Theory",
*J. Phys. Chem. A* **110**, 9212-9218 (2006).
"""

import os

import ase.io
import numpy as np

from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# === Constants ==========================================================
HA_TO_EV = 27.211386245988

# === Load Q-TTF-Q structure =============================================
xyzfile = os.path.join(os.path.dirname(__file__), "complex.xyz")
atoms_ref = ase.io.read(xyzfile)

# === Region assignment ==================================================
# The molecule has approximate C2 symmetry about the central C=C bond.
# Left region  (0): atoms 0-6, 18-23  (13 atoms, x < 0)
# Right region (1): atoms 7-17, 24-25 (13 atoms, x > 0)
#
#   atom indices:  0  1  2  3  4  5  6 | 7  8  9 10 11 12 13 14 15 16 17 | 18 19 20 21 22 23 | 24 25
#   region:        0  0  0  0  0  0  0 | 1  1  1  1  1  1  1  1  1  1  1 |  0  0  0  0  0  0 |  1  1
REGION_MASK = [0] * 24 + [1] * 10

# === Set up AIMNet2 calculator ==========================================
TOTAL_CHARGE = 0  # radical anion
base_calc = AIMNet2Calculator("aimnet2_2025")
d0 = [0.0, 0.0]
d1 = [1.0, -1.0]
# === Part 1: Single-point analysis at the reference geometry ============
print("=" * 72)
print("Part 1: Single-point energies at the reference (XYZ) geometry")
print("=" * 72)

# Ground state (unconstrained)
calc_gs = AIMNet2ASE(base_calc, charge=TOTAL_CHARGE)
atoms_sp = atoms_ref.copy()
atoms_sp.calc = calc_gs
e_gs_sp = atoms_sp.get_potential_energy()  # eV (ASE default)

# Diabatic state 1: Q^- on left
calc_d1 = AIMNet2ASE(base_calc, charge=TOTAL_CHARGE, region_mask=REGION_MASK, region_charges=d0)
atoms_sp1 = atoms_ref.copy()
atoms_sp1.calc = calc_d1
e_d1_sp = atoms_sp1.get_potential_energy()

# Diabatic state 2: Q^- on right
calc_d2 = AIMNet2ASE(base_calc, charge=TOTAL_CHARGE, region_mask=REGION_MASK, region_charges=d1)
atoms_sp2 = atoms_ref.copy()
atoms_sp2.calc = calc_d2
e_d2_sp = atoms_sp2.get_potential_energy()

# Diabatic coupling at reference geometry (Eq. 8 of CQEq paper)

h12_sp = np.sqrt((e_d1_sp - e_gs_sp) * (e_d2_sp - e_gs_sp))

print(f"  E_gs  (adiabatic)         = {e_gs_sp:12.6f} eV")
print(f"  E_1   (Q^- left)          = {e_d1_sp:12.6f} eV")
print(f"  E_2   (Q^- right)         = {e_d2_sp:12.6f} eV")
print(f"  |H_12| (ref geom)         = {h12_sp:12.6f} eV")
print()
