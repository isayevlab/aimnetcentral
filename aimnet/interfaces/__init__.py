"""Wrappers exposing AIMNet2 to external MD/QM packages.

Public integrations:
    * ASE        -- aimnet.calculators.aimnet2ase.AIMNet2ASE
    * pysisyphus -- aimnet.calculators.aimnet2pysis.AIMNet2Pysis

Work-in-progress integrations live in this package but are intentionally
not re-exported here until they are functional end-to-end. See:
    * gromacs.py -- TorchScript export wrapper for GROMACS NNPot
                    (parked: blocked by upstream TorchScript-export issues
                    in the core AIMNet2 module and DFTD3 autograd; see
                    docs/external/gromacs.md and the planning doc at
                    docs/superpowers/plans/2026-04-26-torchscript-export.md)
"""

__all__: list[str] = []
