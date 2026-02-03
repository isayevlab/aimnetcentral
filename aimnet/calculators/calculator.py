import math
import warnings
from typing import Any, ClassVar, Literal

import torch
from nvalchemiops.neighborlist import neighbor_list
from nvalchemiops.neighborlist.neighbor_utils import NeighborOverflowError
from torch import Tensor, nn

from aimnet.models.base import load_model
from aimnet.modules import DFTD3, LRCoulomb

from .model_registry import get_model_path


class AdaptiveNeighborList:
    """Adaptive neighbor list with automatic buffer sizing.

    Wraps nvalchemiops.neighborlist.neighbor_list with automatic max_neighbors adjustment.
    Maintains ~75% utilization to balance memory and recomputation.

    Parameters
    ----------
    cutoff : float
        Cutoff distance for neighbor detection in Angstroms.
    density : float, optional
        Initial atomic density estimate for allocation sizing.
        Used to compute initial max_neighbors as density * (4/3 * pi * cutoff^3).
        Default is 0.2.
    target_utilization : float, optional
        Target ratio of actual neighbors to allocated max_neighbors.
        Default is 0.75 (75% utilization).

    Attributes
    ----------
    cutoff : float
        Cutoff distance for neighbor detection.
    target_utilization : float
        Target ratio of actual to allocated neighbors.
    max_neighbors : int
        Current maximum neighbor allocation (rounded to 16).
    """

    def __init__(
        self,
        cutoff: float,
        density: float = 0.2,
        target_utilization: float = 0.75,
    ) -> None:
        self.cutoff = cutoff
        self.target_utilization = target_utilization
        sphere_volume = 4 / 3 * math.pi * cutoff**3
        self.max_neighbors = self._round_to_16(int(density * sphere_volume))

    @staticmethod
    def _round_to_16(n: int) -> int:
        """Round up to the next multiple of 16 for memory alignment."""
        return ((n + 15) // 16) * 16

    def __call__(
        self,
        positions: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        batch_idx: Tensor | None = None,
        fill_value: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute neighbor list with automatic buffer adjustment.

        Parameters
        ----------
        positions : Tensor
            Atomic coordinates, shape (N, 3).
        cell : Tensor | None
            Unit cell vectors, shape (num_systems, 3, 3). None for non-periodic.
        pbc : Tensor | None
            Periodic boundary conditions, shape (num_systems, 3). None for non-periodic.
        batch_idx : Tensor | None
            Batch index for each atom, shape (N,). None for single system.
        fill_value : int | None
            Fill value for padding. Default is N (number of atoms).

        Returns
        -------
        nbmat : Tensor
            Neighbor indices, shape (N, actual_max_neighbors).
        num_neighbors : Tensor
            Number of neighbors per atom, shape (N,).
        shifts : Tensor | None
            Integer unit cell shifts for PBC, shape (N, actual_max_neighbors, 3).
            None for non-periodic systems.
        """
        N = positions.shape[0]
        if fill_value is None:
            fill_value = N
        _pbc = cell is not None

        while True:
            try:
                if _pbc:
                    nbmat, num_neighbors, shifts = neighbor_list(
                        positions=positions,
                        cutoff=self.cutoff,
                        cell=cell,
                        pbc=pbc,
                        batch_idx=batch_idx,
                        max_neighbors=self.max_neighbors,
                        half_fill=False,
                        fill_value=fill_value,
                    )
                else:
                    nbmat, num_neighbors = neighbor_list(
                        positions=positions,
                        cutoff=self.cutoff,
                        batch_idx=batch_idx,
                        max_neighbors=self.max_neighbors,
                        half_fill=False,
                        fill_value=fill_value,
                    )
                    shifts = None
            except NeighborOverflowError:
                # Increase buffer by 1.5x and retry
                self.max_neighbors = self._round_to_16(int(self.max_neighbors * 1.5))
                continue

            # Get actual max neighbors from result
            actual_max = int(num_neighbors.max().item())

            # Adjust buffer if under-utilized (shrink at 2/3 of target for hysteresis)
            # Use 2/3 threshold to prevent thrashing from small fluctuations
            if actual_max < (2 / 3) * self.target_utilization * self.max_neighbors:
                new_max = self._round_to_16(int(actual_max / self.target_utilization))
                self.max_neighbors = max(new_max, 16)  # Ensure minimum of 16

            # Trim to actual max neighbors
            actual_nnb = max(1, actual_max)
            nbmat = nbmat[:, :actual_nnb]
            if shifts is not None:
                shifts = shifts[:, :actual_nnb]

            return nbmat, num_neighbors, shifts


class AIMNet2Calculator:
    """Generic AIMNet2 calculator.

    A helper class to load AIMNet2 models and perform inference.

    Parameters
    ----------
    model : str | nn.Module
        Model name (from registry), path to model file, or nn.Module instance.
    nb_threshold : int
        Threshold for neighbor list batching. Molecules larger than this use
        flattened processing. Default is 120.
    needs_coulomb : bool | None
        Whether to add external Coulomb module. If None (default), determined
        from model metadata. If True/False, overrides metadata.
    needs_dispersion : bool | None
        Whether to add external DFTD3 module. If None (default), determined
        from model metadata. If True/False, overrides metadata.
    device : str | None
        Device to run the model on ("cuda", "cpu", or specific like "cuda:0").
        If None (default), auto-detects CUDA availability.
    compile_model : bool
        Whether to compile the model with torch.compile(). Default is False.
    compile_kwargs : dict | None
        Additional keyword arguments to pass to torch.compile(). Default is None.
    train : bool
        Whether to enable training mode. Default is False (inference mode).
        When False, all model parameters have requires_grad=False, which
        improves torch.compile compatibility and reduces memory usage.
        Set to True only when training the model.

    Attributes
    ----------
    model : nn.Module
        The loaded AIMNet2 model.
    device : str
        Device the model is running on ("cuda" or "cpu").
    cutoff : float
        Short-range cutoff distance in Angstroms.
    cutoff_lr : float | None
        Long-range cutoff distance, or None if no LR modules.
    external_coulomb : LRCoulomb | None
        External Coulomb module if attached.
    external_dftd3 : DFTD3 | None
        External DFTD3 module if attached.

    Notes
    -----
    External LR module behavior:

    - For file-loaded models (str): metadata is loaded from file
    - For nn.Module: metadata is read from model.metadata attribute if available
    - Explicit flags (needs_coulomb, needs_dispersion) override metadata
    - If no metadata and no explicit flags, no external LR modules are added
    """

    keys_in: ClassVar[dict[str, torch.dtype]] = {"coord": torch.float, "numbers": torch.int, "charge": torch.float}
    keys_in_optional: ClassVar[dict[str, torch.dtype]] = {
        "mult": torch.float,
        "mol_idx": torch.int,
        "nbmat": torch.int,
        "nbmat_lr": torch.int,
        "nb_pad_mask": torch.bool,
        "nb_pad_mask_lr": torch.bool,
        "shifts": torch.float,
        "shifts_lr": torch.float,
        "cell": torch.float,
    }
    keys_out: ClassVar[list[str]] = ["energy", "charges", "forces", "hessian", "stress"]
    atom_feature_keys: ClassVar[list[str]] = ["coord", "numbers", "charges", "forces"]

    def __init__(
        self,
        model: str | nn.Module = "aimnet2",
        nb_threshold: int = 120,
        needs_coulomb: bool | None = None,
        needs_dispersion: bool | None = None,
        device: str | None = None,
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
        train: bool = False,
    ):
        # Device selection: use provided or auto-detect
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.external_coulomb: LRCoulomb | None = None
        self.external_dftd3: DFTD3 | None = None
        # Default cutoffs for LR modules
        self._default_dsf_cutoff = 15.0
        self._default_dftd3_cutoff = 15.0
        self._default_dftd3_smoothing = 0.2

        # Load model and get metadata
        metadata: dict | None = None
        if isinstance(model, str):
            p = get_model_path(model)
            self.model, metadata = load_model(p, device=self.device)
            self.cutoff = metadata["cutoff"]
        elif isinstance(model, nn.Module):
            self.model = model.to(self.device)
            self.cutoff = getattr(self.model, "cutoff", 5.0)
            metadata = getattr(self.model, "_metadata", None)
        else:
            raise TypeError("Invalid model type/name.")

        # Compile model if requested
        if compile_model:
            kwargs = compile_kwargs or {}
            self.model = torch.compile(self.model, **kwargs)

        # Resolve final flags (explicit overrides metadata)
        final_needs_coulomb = (
            needs_coulomb
            if needs_coulomb is not None
            else (metadata.get("needs_coulomb", False) if metadata is not None else False)
        )
        final_needs_dispersion = (
            needs_dispersion
            if needs_dispersion is not None
            else (metadata.get("needs_dispersion", False) if metadata is not None else False)
        )

        # Set up external Coulomb if needed
        if final_needs_coulomb:
            sr_embedded = (
                metadata.get("coulomb_mode") == "sr_embedded" if metadata is not None else False
            )
            # For PBC, user can switch to DSF/Ewald via set_lrcoulomb_method()
            # When sr_embedded=True: model has SRCoulomb which subtracts SR, so external
            # should compute FULL (subtract_sr=False) to give: (NN - SR) + FULL = NN + LR
            # When sr_embedded=False: model has no SR embedded, so external should compute
            # LR only (subtract_sr=True) to avoid double-counting
            self.external_coulomb = LRCoulomb(
                key_in="charges",
                key_out="energy",
                method="simple",
                rc=metadata.get("coulomb_sr_rc", 4.6) if metadata is not None else 4.6,
                envelope=metadata.get("coulomb_sr_envelope", "exp") if metadata is not None else "exp",
                subtract_sr=not sr_embedded,
            )
            self.external_coulomb = self.external_coulomb.to(self.device)

        # Set up external DFTD3 if needed
        if final_needs_dispersion:
            d3_params = metadata.get("d3_params") if metadata else None
            if d3_params is None:
                raise ValueError(
                    "needs_dispersion=True but d3_params not found in metadata. "
                    "Provide d3_params in model metadata or set needs_dispersion=False."
                )
            self.external_dftd3 = DFTD3(
                s8=d3_params["s8"],
                a1=d3_params["a1"],
                a2=d3_params["a2"],
                s6=d3_params.get("s6", 1.0),
            )
            self.external_dftd3 = self.external_dftd3.to(self.device)

        # Determine if model has long-range modules (embedded or external)
        has_embedded_lr = metadata.get("has_embedded_lr", False) if metadata is not None else False
        self.lr = (
            hasattr(self.model, "cutoff_lr")
            or self.external_coulomb is not None
            or self.external_dftd3 is not None
            or has_embedded_lr
        )
        # Set cutoff_lr based on model attribute or external modules
        if hasattr(self.model, "cutoff_lr"):
            self.cutoff_lr = getattr(self.model, "cutoff_lr", float("inf"))
        elif self.external_coulomb is not None:
            # For "simple" method, use inf (all pairs). For DSF, use dsf_cutoff.
            if self.external_coulomb.method == "simple":
                self.cutoff_lr = float("inf")
            else:
                self.cutoff_lr = self._default_dsf_cutoff
        elif self.external_dftd3 is not None:
            self.cutoff_lr = self._default_dftd3_cutoff
        elif has_embedded_lr:
            # Embedded LR modules (D3TS, SRCoulomb) need nbmat_lr
            self.cutoff_lr = self._default_dftd3_cutoff
        else:
            self.cutoff_lr = None
        self.nb_threshold = nb_threshold

        # Create adaptive neighbor list instances
        self._nblist = AdaptiveNeighborList(cutoff=self.cutoff)

        # Track separate cutoffs for LR modules
        self._coulomb_cutoff: float | None = None
        self._dftd3_cutoff: float = self._default_dftd3_cutoff
        if self.external_coulomb is not None:
            if self.external_coulomb.method == "simple":
                self._coulomb_cutoff = float("inf")
            elif self.external_coulomb.method == "ewald":
                self._coulomb_cutoff = None  # Ewald manages its own cutoff
            else:
                self._coulomb_cutoff = self.external_coulomb.dsf_rc
        if self.external_dftd3 is not None:
            self._dftd3_cutoff = self.external_dftd3.smoothing_off

        # Create long-range neighbor list(s) if LR modules present
        self._nblist_lr: AdaptiveNeighborList | None = None
        self._nblist_dftd3: AdaptiveNeighborList | None = None
        self._nblist_coulomb: AdaptiveNeighborList | None = None
        self._update_lr_nblists()

        # indicator if input was flattened
        self._batch = None
        self._max_mol_size: int = 0
        # placeholder for tensors that require grad
        self._saved_for_grad = {}
        # set flag of current Coulomb method
        self._coulomb_method: str | None = None
        if self.external_coulomb is not None:
            self._coulomb_method = self.external_coulomb.method
        elif self._has_embedded_coulomb():
            # Legacy models have embedded Coulomb with "simple" method
            self._coulomb_method = "simple"

        # Set training mode (default False for inference)
        self._train = train
        self.model.train(train)
        if not train:
            # Disable gradients on all parameters for inference mode
            for param in self.model.parameters():
                param.requires_grad_(False)
            if self.external_coulomb is not None:
                for param in self.external_coulomb.parameters():
                    param.requires_grad_(False)
            if self.external_dftd3 is not None:
                for param in self.external_dftd3.parameters():
                    param.requires_grad_(False)

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    @property
    def has_external_coulomb(self) -> bool:
        """Check if calculator has external Coulomb module attached.

        Returns True for new-format models that were trained with Coulomb
        and have it externalized. For legacy models, Coulomb is embedded
        in the model itself, so this returns False.
        """
        return self.external_coulomb is not None

    @property
    def has_external_dftd3(self) -> bool:
        """Check if calculator has external DFTD3 module attached.

        Returns True for new-format models that were trained with DFTD3/D3BJ
        dispersion and have it externalized. For legacy models or D3TS models,
        dispersion is embedded in the model itself, so this returns False.
        """
        return self.external_dftd3 is not None

    @property
    def coulomb_method(self) -> str | None:
        """Get the current Coulomb method.

        Returns
        -------
        str | None
            One of "simple", "dsf", "ewald", or None if no external Coulomb.
            For legacy models with embedded Coulomb, returns None.
        """
        if self.external_coulomb is not None:
            return self.external_coulomb.method
        return None

    @property
    def coulomb_cutoff(self) -> float | None:
        """Get the current Coulomb cutoff distance.

        Returns
        -------
        float | None
            The cutoff distance for Coulomb calculations, or None if not applicable.
            For "simple" method, this is inf. For "ewald", this is None.
            Use set_lrcoulomb_method() to change.
        """
        return self._coulomb_cutoff

    @property
    def dftd3_cutoff(self) -> float:
        """Get the current DFTD3 cutoff distance.

        Returns
        -------
        float
            The cutoff distance for DFTD3 calculations in Angstroms.
        """
        return self._dftd3_cutoff

    def _has_embedded_dispersion(self) -> bool:
        """Check if model has embedded dispersion (not externalized).

        Uses model metadata when available, otherwise returns False (unknown).

        Returns
        -------
        bool
            True if model has embedded dispersion module (D3TS or legacy DFTD3).
        """
        meta = getattr(self.model, "_metadata", None)
        if meta is None:
            return False  # Unknown, assume no embedded dispersion

        # New format: Check for embedded D3TS via has_embedded_lr
        # If has_embedded_lr=True and coulomb_mode != "sr_embedded", it's D3TS
        if meta.get("has_embedded_lr", False):
            coulomb_mode = meta.get("coulomb_mode", "none")
            if coulomb_mode != "sr_embedded":
                return True  # Must be D3TS (embedded dispersion)

        # Legacy format: If needs_dispersion=False and d3_params exist, dispersion is embedded
        # (legacy JIT models have dispersion embedded)
        return not meta.get("needs_dispersion", False) and meta.get("d3_params") is not None

    def _has_embedded_coulomb(self) -> bool:
        """Check if model has embedded Coulomb (not externalized).

        Uses model metadata when available, otherwise returns False (unknown).

        Returns
        -------
        bool
            True if model has embedded Coulomb module.
        """
        meta = getattr(self.model, "_metadata", None)
        if meta is None:
            return False  # Unknown, assume no embedded Coulomb
        # If needs_coulomb=False and coulomb_mode is not "none", Coulomb is embedded
        # (legacy JIT models have full Coulomb embedded)
        return not meta.get("needs_coulomb", False) and meta.get("coulomb_mode", "none") != "none"

    def _should_use_separate_nblist(self, cutoff1: float, cutoff2: float) -> bool:
        """Check if two cutoffs differ enough to warrant separate neighbor lists.

        Parameters
        ----------
        cutoff1 : float
            First cutoff distance.
        cutoff2 : float
            Second cutoff distance.

        Returns
        -------
        bool
            True if cutoffs differ by more than 20%, False otherwise.
        """
        # Handle edge cases
        if cutoff1 <= 0 or cutoff2 <= 0:
            return False
        if not math.isfinite(cutoff1) or not math.isfinite(cutoff2):
            return False
        ratio = max(cutoff1, cutoff2) / min(cutoff1, cutoff2)
        return ratio > 1.2

    def _update_lr_nblists(self) -> None:
        """Update long-range neighbor list instances based on current cutoffs.

        Creates separate neighbor lists for DFTD3 and Coulomb if their cutoffs
        differ by more than 20%. Otherwise, uses a single shared neighbor list.
        Ewald uses its own internal neighbor list and ignores cutoffs.
        """
        if not self.lr:
            self._nblist_lr = None
            self._nblist_dftd3 = None
            self._nblist_coulomb = None
            return

        has_dftd3 = self.external_dftd3 is not None or self._has_embedded_dispersion()
        has_coulomb = self.external_coulomb is not None or self._has_embedded_coulomb()

        # Determine effective cutoffs (None means no neighbor list needed for that module)
        dftd3_cutoff = self._dftd3_cutoff if has_dftd3 else 0.0
        coulomb_cutoff = self._coulomb_cutoff if has_coulomb and self._coulomb_cutoff is not None else 0.0

        # Check if we need separate neighbor lists (both finite and differ by >20%)
        if (
            has_dftd3
            and has_coulomb
            and math.isfinite(dftd3_cutoff)
            and math.isfinite(coulomb_cutoff)
            and coulomb_cutoff > 0
            and self._should_use_separate_nblist(dftd3_cutoff, coulomb_cutoff)
        ):
            # Use separate neighbor lists
            self._nblist_dftd3 = AdaptiveNeighborList(cutoff=dftd3_cutoff)
            self._nblist_coulomb = AdaptiveNeighborList(cutoff=coulomb_cutoff)
            self._nblist_lr = None
            return

        # Use single shared neighbor list with max cutoff
        max_cutoff = 0.0
        if has_dftd3 and math.isfinite(dftd3_cutoff):
            max_cutoff = max(max_cutoff, dftd3_cutoff)
        if has_coulomb:
            if coulomb_cutoff == float("inf"):
                # Simple Coulomb needs all pairs
                self._nblist_lr = AdaptiveNeighborList(cutoff=1e6)
                self._nblist_dftd3 = None
                self._nblist_coulomb = None
                return
            if math.isfinite(coulomb_cutoff) and coulomb_cutoff > 0:
                max_cutoff = max(max_cutoff, coulomb_cutoff)

        if max_cutoff > 0:
            self._nblist_lr = AdaptiveNeighborList(cutoff=max_cutoff)
        else:
            self._nblist_lr = None
        self._nblist_dftd3 = None
        self._nblist_coulomb = None

    def set_lrcoulomb_method(
        self,
        method: Literal["simple", "dsf", "ewald"],
        cutoff: float = 15.0,
        dsf_alpha: float = 0.2,
        ewald_accuracy: float = 1e-8,
    ):
        """Set the long-range Coulomb method.

        Parameters
        ----------
        method : str
            One of "simple", "dsf", or "ewald".
        cutoff : float
            Cutoff distance for DSF neighbor list. Default is 15.0.
            Not used for Ewald (which computes cutoffs from accuracy).
        dsf_alpha : float
            Alpha parameter for DSF method. Default is 0.2.
        ewald_accuracy : float
            Target accuracy for Ewald summation. Controls the real-space
            and reciprocal-space cutoffs. Lower values give higher accuracy
            but require more computation. Default is 1e-8.

            The Ewald cutoffs are computed as:
            - eta = (V^2 / N)^(1/6) / sqrt(2*pi)
            - cutoff_real = sqrt(-2 * ln(accuracy)) * eta
            - cutoff_recip = sqrt(-2 * ln(accuracy)) / eta

        Notes
        -----
        For new-format models with external Coulomb, this updates the external module.
        For legacy models with embedded Coulomb, a warning is issued as those modules
        cannot be modified at runtime.
        """
        if method not in ("simple", "dsf", "ewald"):
            raise ValueError(f"Invalid method: {method}")

        # Warn if model has embedded Coulomb (legacy models)
        if self._has_embedded_coulomb() and self.external_coulomb is None:
            warnings.warn(
                "Model has embedded Coulomb module (legacy format). "
                "set_lrcoulomb_method() only affects external Coulomb modules. "
                "For legacy models, the Coulomb method cannot be changed at runtime.",
                stacklevel=2,
            )

        # Update external LRCoulomb module if present
        if self.external_coulomb is not None:
            self.external_coulomb.method = method
            if method == "dsf":
                self.external_coulomb.dsf_alpha = dsf_alpha
                self.external_coulomb.dsf_rc = cutoff
            elif method == "ewald":
                self.external_coulomb.ewald_accuracy = ewald_accuracy

        # Update _coulomb_cutoff based on method
        if method == "simple":
            self._coulomb_cutoff = float("inf")
        elif method == "dsf":
            self._coulomb_cutoff = cutoff
        elif method == "ewald":
            self._coulomb_cutoff = None  # Ewald manages its own real-space cutoff

        # Update cutoff_lr for backward compatibility
        if self._coulomb_cutoff is not None:
            self.cutoff_lr = self._coulomb_cutoff
        else:
            # Ewald - use DFTD3 cutoff if available, else None
            self.cutoff_lr = self._dftd3_cutoff if self.external_dftd3 is not None else None

        self._coulomb_method = method
        self._update_lr_nblists()

    def set_lr_cutoff(self, cutoff: float) -> None:
        """Set the unified long-range cutoff for all LR modules.

        Parameters
        ----------
        cutoff : float
            Cutoff distance in Angstroms for LR neighbor lists.

        Notes
        -----
        This updates both _coulomb_cutoff and _dftd3_cutoff.
        Ewald uses its own internal neighbor list and ignores this cutoff.
        """
        # Update both cutoffs (but not for ewald which manages its own)
        if self._coulomb_method != "ewald":
            self._coulomb_cutoff = cutoff
        self._dftd3_cutoff = cutoff
        self.cutoff_lr = cutoff
        self._update_lr_nblists()

    def set_dftd3_cutoff(self, cutoff: float | None = None, smoothing_fraction: float | None = None) -> None:
        """Set DFTD3 cutoff and smoothing.

        Parameters
        ----------
        cutoff : float | None
            Cutoff distance in Angstroms for DFTD3 calculation.
            Default is _default_dftd3_cutoff (15.0).
        smoothing_fraction : float | None
            Fraction of cutoff used as smoothing width.
            Default is _default_dftd3_smoothing (0.2).

        Notes
        -----
        This method only affects external DFTD3 modules attached to
        new-format models. For legacy models with embedded DFTD3,
        the smoothing is fixed.

        Updates _dftd3_cutoff and rebuilds neighbor lists.
        """
        if cutoff is None:
            cutoff = self._default_dftd3_cutoff
        if smoothing_fraction is None:
            smoothing_fraction = self._default_dftd3_smoothing

        self._dftd3_cutoff = cutoff
        if self.external_dftd3 is not None:
            self.external_dftd3.set_smoothing(cutoff, smoothing_fraction)
        self._update_lr_nblists()

    def eval(self, data: dict[str, Any], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        data = self.prepare_input(data)

        if hessian and "mol_idx" in data and data["mol_idx"][-1] > 0:
            raise NotImplementedError("Hessian calculation is not supported for multiple molecules")
        data = self.set_grad_tensors(data, forces=forces, stress=stress, hessian=hessian)
        with torch.jit.optimized_execution(False):  # type: ignore
            data = self.model(data)
        # Run external modules if present
        data = self._run_external_modules(data)
        data = self.get_derivatives(data, forces=forces, stress=stress, hessian=hessian)
        data = self.process_output(data)
        return data

    def _run_external_modules(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run external Coulomb and DFTD3 modules if attached."""
        if self.external_coulomb is not None:
            data = self.external_coulomb(data)
        if self.external_dftd3 is not None:
            data = self.external_dftd3(data)
        return data

    def prepare_input(self, data: dict[str, Any]) -> dict[str, Tensor]:
        data = self.to_input_tensors(data)
        data = self.mol_flatten(data)
        if data.get("cell") is not None:
            if self._coulomb_method == "simple":
                warnings.warn("Switching to DSF Coulomb for PBC", stacklevel=1)
                self.set_lrcoulomb_method("dsf")
        if data["coord"].ndim == 2:
            # Skip neighbor list calculation if already provided
            if "nbmat" not in data:
                data = self.make_nbmat(data)
            data = self.pad_input(data)
        return data

    def process_output(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if data["coord"].ndim == 2:
            data = self.unpad_output(data)
        data = self.mol_unflatten(data)
        data = self.keep_only(data)
        return data

    def to_input_tensors(self, data: dict[str, Any]) -> dict[str, Tensor]:
        ret = {}
        for k in self.keys_in:
            if k not in data:
                raise KeyError(f"Missing key {k} in the input data")
            # Detach from computation graph to prevent gradient accumulation
            ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k]).detach()
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in_optional[k]).detach()
        # Ensure all tensors have at least 1D shape for consistent batch processing
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)
        return ret

    def mol_flatten(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Flatten the input data for multiple molecules.
        Will not flatten for batched input and molecule size below threshold.
        """
        ndim = data["coord"].ndim
        if ndim == 2:
            self._batch = None
            if "mol_idx" not in data:
                data["mol_idx"] = torch.zeros(data["coord"].shape[0], dtype=torch.long, device=self.device)
                self._max_mol_size = data["coord"].shape[0]
            elif data["mol_idx"][-1] == 0:
                self._max_mol_size = len(data["mol_idx"])
            else:
                self._max_mol_size = data["mol_idx"].unique(return_counts=True)[1].max().item()

        elif ndim == 3:
            B, N = data["coord"].shape[:2]
            # Force flattening for PBC (cell present) to ensure make_nbmat computes proper neighbor lists with shifts
            if self.nb_threshold < N or self.device == "cpu" or data.get("cell") is not None:
                self._batch = B
                data["mol_idx"] = torch.repeat_interleave(
                    torch.arange(0, B, device=self.device), torch.full((B,), N, device=self.device)
                )
                for k, v in data.items():
                    if k in self.atom_feature_keys:
                        data[k] = v.flatten(0, 1)
            else:
                self._batch = None
            self._max_mol_size = N
        return data

    def mol_unflatten(self, data: dict[str, Tensor], batch=None) -> dict[str, Tensor]:
        batch = batch if batch is not None else self._batch
        if batch is not None:
            for k, v in data.items():
                if k in self.atom_feature_keys:
                    data[k] = v.view(batch, -1, *v.shape[1:])
        return data

    def make_nbmat(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        assert self._max_mol_size > 0, "Molecule size is not set"

        # Prepare batch_idx from mol_idx
        mol_idx = data.get("mol_idx")

        if "cell" in data and data["cell"] is not None:
            data["coord"] = move_coord_to_cell(data["coord"], data["cell"], mol_idx)
            cell = data["cell"]
        else:
            cell = None

        N = data["coord"].shape[0]
        _pbc = cell is not None
        batch_idx = mol_idx.to(torch.int32) if mol_idx is not None else None

        # Prepare cell and pbc tensors for nvalchemiops
        if _pbc:
            if cell.ndim == 2:
                cell_batched = cell.unsqueeze(0)  # (1, 3, 3)
            else:
                cell_batched = cell  # (num_systems, 3, 3)
            num_systems = cell_batched.shape[0]
            pbc = torch.tensor([[True, True, True]] * num_systems, dtype=torch.bool, device=cell.device)
        else:
            cell_batched = None
            pbc = None

        # Short-range neighbors (always)
        nbmat1, _, shifts1 = self._nblist(
            positions=data["coord"],
            cell=cell_batched,
            pbc=pbc,
            batch_idx=batch_idx,
            fill_value=N,
        )

        nbmat1, shifts1 = _add_padding_row(nbmat1, shifts1, N)
        data["nbmat"] = nbmat1
        if cell is not None:
            assert shifts1 is not None
            data["shifts"] = shifts1

        # Unified neighbor list when LR module cutoffs are similar
        if self._nblist_lr is not None:
            if self._coulomb_cutoff == float("inf"):
                self._nblist_lr.max_neighbors = N
            nbmat_lr, _, shifts_lr = self._nblist_lr(
                positions=data["coord"],
                cell=cell_batched,
                pbc=pbc,
                batch_idx=batch_idx,
                fill_value=N,
            )
            nbmat_lr, shifts_lr = _add_padding_row(nbmat_lr, shifts_lr, N)

            # All LR modules share the same neighbor list when cutoffs are similar
            data["nbmat_lr"] = nbmat_lr
            data["nbmat_coulomb"] = nbmat_lr
            data["nbmat_dftd3"] = nbmat_lr
            if cell is not None and shifts_lr is not None:
                data["shifts_lr"] = shifts_lr
                data["shifts_coulomb"] = shifts_lr
                data["shifts_dftd3"] = shifts_lr
        else:
            if self._nblist_coulomb is not None:
                if self._coulomb_cutoff == float("inf"):
                    self._nblist_coulomb.max_neighbors = N
                nbmat_coulomb, _, shifts_coulomb = self._nblist_coulomb(
                    positions=data["coord"],
                    cell=cell_batched,
                    pbc=pbc,
                    batch_idx=batch_idx,
                    fill_value=N,
                )
                nbmat_coulomb, shifts_coulomb = _add_padding_row(nbmat_coulomb, shifts_coulomb, N)
                data["nbmat_coulomb"] = nbmat_coulomb
                # Set nbmat_lr for backward compatibility with code expecting unified LR neighbor list
                data["nbmat_lr"] = nbmat_coulomb
                if cell is not None and shifts_coulomb is not None:
                    data["shifts_coulomb"] = shifts_coulomb
                    data["shifts_lr"] = shifts_coulomb

                if self._nblist_dftd3 is not None:
                    nbmat_dftd3, _, shifts_dftd3 = self._nblist_dftd3(
                        positions=data["coord"],
                        cell=cell_batched,
                        pbc=pbc,
                        batch_idx=batch_idx,
                        fill_value=N,
                    )
                    nbmat_dftd3, shifts_dftd3 = _add_padding_row(nbmat_dftd3, shifts_dftd3, N)
                    data["nbmat_dftd3"] = nbmat_dftd3
                    if cell is not None and shifts_dftd3 is not None:
                        data["shifts_dftd3"] = shifts_dftd3

            elif self._nblist_dftd3 is not None:
                # DFTD3-only configuration: populate nbmat_lr for backward compatibility
                nbmat_dftd3, _, shifts_dftd3 = self._nblist_dftd3(
                    positions=data["coord"],
                    cell=cell_batched,
                    pbc=pbc,
                    batch_idx=batch_idx,
                    fill_value=N,
                )
                nbmat_dftd3, shifts_dftd3 = _add_padding_row(nbmat_dftd3, shifts_dftd3, N)
                data["nbmat_dftd3"] = nbmat_dftd3
                data["nbmat_lr"] = nbmat_dftd3
                if cell is not None and shifts_dftd3 is not None:
                    data["shifts_dftd3"] = shifts_dftd3
                    data["shifts_lr"] = shifts_dftd3

        return data

    def pad_input(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        N = data["nbmat"].shape[0]
        data["mol_idx"] = maybe_pad_dim0(data["mol_idx"], N, value=data["mol_idx"][-1].item())
        for k in ("coord", "numbers"):
            if k in data:
                data[k] = maybe_pad_dim0(data[k], N)
        return data

    def unpad_output(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        N = data["nbmat"].shape[0] - 1
        for k, v in data.items():
            if k in self.atom_feature_keys:
                data[k] = maybe_unpad_dim0(v, N)
        return data

    def set_grad_tensors(self, data: dict[str, Tensor], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        self._saved_for_grad = {}
        if forces or hessian:
            data["coord"].requires_grad_(True)
            self._saved_for_grad["coord"] = data["coord"]
        if stress:
            assert "cell" in data and data["cell"] is not None, "Stress calculation requires cell"
            scaling = torch.eye(3, requires_grad=True, dtype=data["cell"].dtype, device=data["cell"].device)
            data["coord"] = data["coord"] @ scaling
            data["cell"] = data["cell"] @ scaling
            self._saved_for_grad["scaling"] = scaling
        return data

    def keep_only(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        ret = {}
        for k, v in data.items():
            if k in self.keys_out or (k.endswith("_std") and k[:-4] in self.keys_out):
                ret[k] = v
        return ret

    def get_derivatives(self, data: dict[str, Tensor], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        # Use stored train mode for create_graph decision
        _create_graph = hessian or self._train
        x = []
        if hessian:
            forces = True
        if forces and ("forces" not in data or (_create_graph and not data["forces"].requires_grad)):
            forces = True
            x.append(self._saved_for_grad["coord"])
        if stress:
            x.append(self._saved_for_grad["scaling"])
        if x:
            tot_energy = data["energy"].sum()
            deriv = torch.autograd.grad(tot_energy, x, create_graph=_create_graph)
            if forces:
                data["forces"] = -deriv[0]
            if stress:
                dedc = deriv[0] if not forces else deriv[1]
                cell = data["cell"].detach()
                if cell.ndim == 2:
                    # Single cell (3, 3)
                    volume = cell.det().abs()
                else:
                    # Batched cells (B, 3, 3) - compute volume for each cell
                    volume = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                data["stress"] = dedc / volume
        if hessian:
            data["hessian"] = self.calculate_hessian(data["forces"], self._saved_for_grad["coord"])
        return data

    @staticmethod
    def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
        # Coord includes padding atom (shape N+1), forces only for real atoms (shape N)
        # Hessian computed only for actual atoms: (N, 3, N, 3)
        hessian = -torch.stack([
            torch.autograd.grad(_f, coord, retain_graph=True)[0] for _f in forces.flatten().unbind()
        ]).view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
        return hessian


def _add_padding_row(
    nbmat: Tensor,
    shifts: Tensor | None,
    N: int,
) -> tuple[Tensor, Tensor | None]:
    """Add padding row to neighbor matrix and shifts.

    Parameters
    ----------
    nbmat : Tensor
        Neighbor matrix, shape (N, max_neighbors).
    shifts : Tensor | None
        Shift vectors for PBC or None, shape (N, max_neighbors, 3).
    N : int
        Number of atoms (used as fill value for padding row).

    Returns
    -------
    tuple[Tensor, Tensor | None]
        Tuple of (nbmat, shifts) with padding row added.
    """
    device = nbmat.device
    dtype = nbmat.dtype
    nnb_max = nbmat.shape[1]
    padding_row = torch.full((1, nnb_max), N, dtype=dtype, device=device)
    nbmat = torch.cat([nbmat, padding_row], dim=0)

    if shifts is not None:
        shifts_padding = torch.zeros((1, nnb_max, 3), dtype=shifts.dtype, device=device)
        shifts = torch.cat([shifts, shifts_padding], dim=0)

    return nbmat, shifts


def maybe_pad_dim0(a: Tensor, N: int, value=0.0) -> Tensor:
    _shape_diff = N - a.shape[0]
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a


def pad_dim0(a: Tensor, value=0.0) -> Tensor:
    shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
    a = torch.nn.functional.pad(a, shapes, mode="constant", value=value)
    return a


def maybe_unpad_dim0(a: Tensor, N: int) -> Tensor:
    _shape_diff = a.shape[0] - N
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = a[:-1]
    return a


def move_coord_to_cell(coord: Tensor, cell: Tensor, mol_idx: Tensor | None = None) -> Tensor:
    """Move coordinates into the periodic cell.

    Parameters
    ----------
    coord : Tensor
        Coordinates tensor, shape (N, 3) or (B, N, 3).
    cell : Tensor
        Cell tensor, shape (3, 3) or (B, 3, 3).
    mol_idx : Tensor | None
        Molecule index for each atom, shape (N,).
        Required for batched cells with flat coordinates.

    Returns
    -------
    Tensor
        Coordinates wrapped into the cell.
    """
    if cell.ndim == 2:
        # Single cell (3, 3)
        cell_inv = torch.linalg.inv(cell)
        coord_f = coord @ cell_inv
        coord_f = coord_f % 1
        return coord_f @ cell
    else:
        # Batched cells (B, 3, 3)
        if coord.ndim == 3:
            # Batched coords (B, N, 3) with batched cells (B, 3, 3)
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            coord_f = torch.bmm(coord, cell_inv)  # (B, N, 3)
            coord_f = coord_f % 1
            return torch.bmm(coord_f, cell)
        else:
            # Flat coords (N_total, 3) with batched cells (B, 3, 3) - need mol_idx
            assert mol_idx is not None, "mol_idx required for batched cells with flat coordinates"
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            # Get cell and cell_inv for each atom
            atom_cell = cell[mol_idx]  # (N_total, 3, 3)
            atom_cell_inv = cell_inv[mol_idx]  # (N_total, 3, 3)
            coord_f = torch.bmm(coord.unsqueeze(1), atom_cell_inv).squeeze(1)  # (N_total, 3)
            coord_f = coord_f % 1
            return torch.bmm(coord_f.unsqueeze(1), atom_cell).squeeze(1)
