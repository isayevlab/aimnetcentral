import copy
import math
import os
import re
import warnings
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast

import torch
from nvalchemiops.neighbors import NeighborOverflowError
from nvalchemiops.torch.neighbors import neighbor_list
from torch import Tensor, nn

from aimnet.models.base import load_model
from aimnet.modules import DFTD3, LRCoulomb
from aimnet.modules.lr import ExternalDerivativeTerms

from .model_registry import get_family_policy, get_model_path, get_registry_model_family

# Sentinel for "attribute did not exist" when snapshotting/restoring instance state.
_SENTINEL = object()


def _sum_optional_tensor(x: Tensor | None, y: Tensor | None) -> Tensor | None:
    """Elementwise sum of two ``Optional[Tensor]`` operands."""
    if x is None:
        return y
    if y is None:
        return x
    return x + y.to(dtype=x.dtype, device=x.device)


def _combine_external_terms(
    a: ExternalDerivativeTerms | None,
    b: ExternalDerivativeTerms | None,
) -> ExternalDerivativeTerms | None:
    """Sum forces and virials of two external derivative terms.

    Both inputs follow the calculator-side contract used by
    :meth:`AIMNet2Calculator.get_derivatives`: ``forces`` add to the
    autograd-derived forces and ``virial`` enters as ``dedc -= virial.mT``.
    DSF Coulomb and DFTD3 both publish detached terms in this convention, so
    combining them is a per-system elementwise sum.
    """
    if a is None:
        return b
    if b is None:
        return a
    return ExternalDerivativeTerms(
        forces=_sum_optional_tensor(a.forces, b.forces),
        virial=_sum_optional_tensor(a.virial, b.virial),
        hessian=_sum_optional_tensor(a.hessian, b.hessian),
    )


def _apply_family_defaults(metadata: Mapping[str, Any], registry_family: str | None) -> dict[str, Any]:
    """Apply calculator-side compatibility defaults for released model families."""
    metadata = dict(metadata)
    if registry_family is not None:
        metadata_family = metadata.get("family")
        if metadata_family is None:
            metadata["family"] = registry_family
        elif metadata_family != registry_family:
            raise ValueError(
                f"Registry family '{registry_family}' does not match model metadata family "
                f"'{metadata_family}'. Refusing to load ambiguous energy scale."
            )

    policy = get_family_policy(metadata.get("family"))

    if policy.supports_charged_systems is not None:
        supports_charged = metadata.get("supports_charged_systems")
        if supports_charged is None:
            metadata["supports_charged_systems"] = policy.supports_charged_systems
        elif supports_charged is not policy.supports_charged_systems:
            raise ValueError(
                f"aimnet2-{policy.family} models must declare "
                f"supports_charged_systems={policy.supports_charged_systems}."
            )

    if policy.posthoc_d3_params is not None and not metadata.get("has_embedded_d3ts", False):
        metadata["needs_dispersion"] = True
        if metadata.get("d3_params") is None:
            metadata["d3_params"] = dict(policy.posthoc_d3_params)

    return metadata


class AdaptiveNeighborList:
    """Adaptive neighbor list with automatic buffer sizing.

    Wraps nvalchemiops.torch.neighbors.neighbor_list with automatic max_neighbors adjustment.
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
                        method="batch_naive",
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
    cache_static : bool
        Whether to cache calculator-built neighbor matrices and explicit
        external DFTD3 terms for repeated static CUDA inputs. Default is False.
        This opt-in cache is limited to exact reuse of the same non-periodic
        2D input tensors and is bypassed for Hessian, stress, training,
        caller-supplied ``mol_idx``, and caller-supplied ``nbmat``.
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
        "pbc": torch.bool,
    }
    keys_out: ClassVar[list[str]] = ["energy", "charges", "spin_charges", "forces", "hessian", "stress"]
    atom_feature_keys: ClassVar[list[str]] = ["coord", "numbers", "charges", "spin_charges", "forces"]
    _constructed_families: ClassVar[set[str]] = set()

    def __init__(
        self,
        model: str | nn.Module = "aimnet2",
        nb_threshold: int = 120,
        needs_coulomb: bool | None = None,
        needs_dispersion: bool | None = None,
        device: str | None = None,
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
        cache_static: bool = False,
        train: bool = False,
        ensemble_member: int = 0,
        revision: str | None = None,
        token: str | None = None,
    ):
        # Device selection: use provided or auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(torch.device(device))
        self.model: nn.Module
        self.external_coulomb: LRCoulomb | None = None
        self.external_dftd3: DFTD3 | None = None
        # Default cutoffs for LR modules
        self._default_dsf_cutoff = 15.0
        self._default_dftd3_cutoff = 15.0
        self._default_dftd3_smoothing = 0.2

        # Load model and get metadata
        metadata: Mapping[str, Any] | None = None
        registry_family: str | None = None
        # Inline org/name pattern — exactly one slash, both segments alphanumeric+._-
        # This avoids importing optional HF deps for ordinary file paths containing slashes.
        _HF_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
        if isinstance(model, str):
            # Check for HF repo ID or local HF-style directory
            # (lazy import to keep safetensors/huggingface_hub optional)
            _is_hf_dir = os.path.isdir(model)
            _looks_like_hf = bool(_HF_ID_RE.match(model))
            if _looks_like_hf or _is_hf_dir:
                try:
                    from aimnet.calculators.hf_hub import is_hf_repo_id, load_from_hf_repo
                except ImportError:
                    raise ImportError(
                        f"Loading from HF repo '{model}' requires optional dependencies. "
                        "Install with: pip install aimnet[hf]"
                    ) from None
                if is_hf_repo_id(model) or _is_hf_dir:
                    _model, metadata = load_from_hf_repo(
                        model,
                        ensemble_member=ensemble_member,
                        device=self.device,
                        revision=revision,
                        token=token,
                    )
                    self.model = _model
                    self.cutoff = metadata["cutoff"]
                else:
                    # _looks_like_hf matched but it's a local file path — fall through
                    if not os.path.isfile(model):
                        registry_family = get_registry_model_family(model)
                    p = get_model_path(model)
                    self.model, metadata = load_model(p, device=self.device)
                    self.cutoff = metadata["cutoff"]
            else:
                if not os.path.isfile(model):
                    registry_family = get_registry_model_family(model)
                p = get_model_path(model)
                self.model, metadata = load_model(p, device=self.device)
                self.cutoff = metadata["cutoff"]
        elif isinstance(model, nn.Module):
            self.model = model.to(self.device)
            self.cutoff = getattr(self.model, "cutoff", 5.0)
            metadata = cast(Mapping[str, Any] | None, getattr(self.model, "metadata", None))
            if metadata is None:
                metadata = cast(Mapping[str, Any] | None, getattr(self.model, "_metadata", None))
        else:
            raise TypeError("Invalid model type/name.")

        if metadata is not None:
            metadata = _apply_family_defaults(metadata, registry_family)
            self.model._metadata = metadata  # type: ignore[assignment]

        # Compile model if requested
        self._was_compiled = bool(compile_model)
        if compile_model:
            kwargs = compile_kwargs or {}
            self.model = cast(nn.Module, torch.compile(self.model, **kwargs))

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
            sr_embedded = metadata.get("coulomb_mode") == "sr_embedded" if metadata is not None else False
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
        self.cache_static = bool(cache_static)
        self._static_nbmat_cache: dict[tuple[Any, ...], tuple[Any, Any, dict[str, Tensor]]] = {}
        self._static_dftd3_cache: dict[tuple[Any, ...], tuple[Any, Any, dict[str, Any]]] = {}
        # Identity of the last species-validated `numbers` tensor, so repeated
        # evals on the same buffer (MD/optimization loops) skip the per-step
        # D2H tolist() + set-diff in _validate_species_and_charge.
        self._species_validation_cache: tuple[tuple[Any, ...], Any, Any] | None = None

        # Create adaptive neighbor list instances
        self._nblist = AdaptiveNeighborList(cutoff=self.cutoff)

        # Track separate cutoffs for LR modules
        self._coulomb_cutoff: float | None = None
        self._dftd3_cutoff: float = self._default_dftd3_cutoff
        if self.external_coulomb is not None:
            if self.external_coulomb.method == "simple":
                self._coulomb_cutoff = float("inf")
            elif self.external_coulomb.method in ("ewald", "pme"):
                self._coulomb_cutoff = None  # Ewald/PME manage their own cutoff
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
        self._batch: int | None = None
        self._max_mol_size: int = 0
        self._static_input_cache_key: tuple[Any, ...] | None = None
        self._static_input_cache_refs: tuple[Any, Any] | None = None
        # placeholder for tensors that require grad
        self._saved_for_grad: dict[str, Tensor] = {}
        # set flag of current Coulomb method
        self._coulomb_method: str | None = None
        if self.external_coulomb is not None:
            self._coulomb_method = self.external_coulomb.method
        elif self._has_embedded_coulomb():
            # Legacy models have embedded Coulomb with "simple" method
            self._coulomb_method = "simple"
        # Bookkeeping for the per-evaluation simple->dsf PBC auto-switch:
        # prepare_input stores the pre-switch Coulomb state here and eval()
        # restores it in its finally block.
        self._pbc_coulomb_restore: dict[str, Any] | None = None
        # Memoized DSF-side state from the auto-switch (incl. warmed-up adaptive
        # neighbor lists) so repeated periodic evals don't rebuild it every step.
        self._auto_dsf_state: dict[str, Any] | None = None
        # One-shot flag for the ignored-multiplicity warning (eval fires inside MD loops).
        self._mult_ignored_checked = False

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

        self._maybe_warn_family_mix((metadata or {}).get("family") if metadata else None)

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.eval(*args, **kwargs)

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """Read-only view of the model's metadata dict.

        Returns a read-only mapping for v2 .pt models, or ``None`` for raw
        ``nn.Module`` inputs that don't carry metadata. Downstream consumers
        should prefer this accessor over reaching into the private
        ``model._metadata`` attribute.
        """
        metadata = getattr(self.model, "_metadata", None)
        return MappingProxyType(metadata) if metadata is not None else None

    def _maybe_warn_family_mix(self, family: str | None) -> None:
        """If multiple distinct families have been constructed in this process,
        emit a one-time UserWarning about energy-scale incompatibility.

        ``family=None`` is the no-op contract — calculators built from raw
        ``nn.Module`` inputs or from .pt files that don't declare ``family``
        in metadata pass ``None`` here and skip both tracking and warning.

        Use Python's standard warnings filters if this warning needs to be silenced.
        """
        if family is None:
            return
        already_warned = family in self._constructed_families
        self._constructed_families.add(family)
        if not already_warned and len(self._constructed_families) > 1:
            warnings.warn(
                f"AIMNet2Calculator instances from different families have been "
                f"constructed in this process: {sorted(self._constructed_families)}. "
                f"Energy targets and reference conventions differ across families "
                f"(e.g. rxn uses a learned shifted-electronic scale, while wb97m-d3 "
                f"and b973c-d3 use different DFT targets). Do not mix or compare "
                f"energies across families.",
                UserWarning,
                stacklevel=2,
            )

    def _maybe_warn_mult_ignored(self, data: dict[str, Any]) -> None:
        """Warn once per instance if ``mult`` != 1 is passed to a closed-shell model.

        ``num_charge_channels=1`` models (e.g. the default aimnet2/wb97m-d3)
        never read ``mult``, while the ASE/pysisyphus/torch-sim wrappers pass it
        unconditionally — without this warning a user requesting an open-shell
        state would silently get the closed-shell result. NSE models
        (num_charge_channels=2) consume ``mult`` and skip the check.

        Use Python's standard warnings filters if this warning needs to be silenced.
        """
        if self._mult_ignored_checked or self.is_nse:
            return
        mult = data.get("mult")
        if mult is None:
            return
        if isinstance(mult, Tensor) and mult.device.type != "cpu":
            # Inspecting a GPU tensor forces a D2H sync; do it at most once per
            # calculator instance (mult is invariant within MD/optimization loops).
            self._mult_ignored_checked = True
            mult_t = mult.detach()
        else:
            mult_t = torch.as_tensor(mult)
        if bool((mult_t != 1).any()):
            self._mult_ignored_checked = True
            warnings.warn(
                f"Input mult={mult_t.flatten().tolist()} is ignored: this model is "
                f"closed-shell (num_charge_channels=1) and does not use spin "
                f"multiplicity, so the result corresponds to the closed-shell state. "
                f"For radicals/open-shell systems use an NSE model, e.g. "
                f"`isayevlab/aimnet2-nse`.",
                UserWarning,
                stacklevel=2,
            )

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
    def is_nse(self) -> bool:
        """Return True if the model supports spin-polarized charges (NSE, num_charge_channels=2)."""
        return getattr(self.model, "num_charge_channels", 1) == 2

    @property
    def coulomb_method(self) -> str | None:
        """Get the current Coulomb method.

        Returns
        -------
        str | None
            One of "simple", "dsf", "ewald", "pme", or None if no external
            Coulomb. For legacy models with embedded Coulomb, returns None.
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
            The cutoff distance for Coulomb calculations, or None if not
            applicable. For ``"simple"`` this is ``inf``; for ``"ewald"`` and
            ``"pme"`` this is ``None`` (cutoff is estimated per call from
            ``ewald_accuracy``). Use ``set_lrcoulomb_method()`` to change.
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

        Reads `has_embedded_d3ts` from metadata when present (the authoritative
        source for new conversions). Falls back to heuristics for legacy .pt
        files that don't carry the explicit flag. Returns False when no metadata
        is available.

        Returns
        -------
        bool
            True if model has embedded dispersion module (D3TS or legacy DFTD3).
        """
        meta = self.metadata
        if meta is None:
            return False  # Unknown, assume no embedded dispersion

        # Authoritative path (new conversions): explicit flag.
        if meta.get("has_embedded_d3ts", False):
            return True

        # Legacy heuristic (pre-explicit-flag .pt files): if has_embedded_lr=True
        # AND coulomb_mode != "sr_embedded", the LR module must be D3TS — but
        # this misses the both-set case (D3TS + SRCoulomb), which is exactly the
        # bug fixed by the explicit flag above. Kept here only as a fallback for
        # legacy files; new conversions take the path above.
        if meta.get("has_embedded_lr", False) and meta.get("coulomb_mode", "none") != "sr_embedded":
            return True

        # Legacy JIT format: needs_dispersion=False + d3_params present means dispersion is embedded.
        return not meta.get("needs_dispersion", False) and meta.get("d3_params") is not None

    def _has_embedded_coulomb(self) -> bool:
        """Check if model has embedded Coulomb (not externalized).

        Uses model metadata when available, otherwise returns False (unknown).

        Returns
        -------
        bool
            True if model has embedded Coulomb module.
        """
        meta = self.metadata
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
        method: Literal["simple", "dsf", "ewald", "pme"],
        cutoff: float = 15.0,
        dsf_alpha: float = 0.2,
        ewald_accuracy: float = 1e-6,
    ):
        """Set the long-range Coulomb method.

        Parameters
        ----------
        method : str
            One of "simple", "dsf", "ewald", or "pme".
        cutoff : float
            Cutoff distance for DSF neighbor list. Default is 15.0.
            Silently ignored for "ewald" and "pme" (which estimate their own
            real-space cutoffs from ``ewald_accuracy``).
        dsf_alpha : float
            Alpha parameter for DSF method. Default is 0.2.
        ewald_accuracy : float
            Target accuracy for Ewald and PME summation. Controls the
            real-space and reciprocal-space cutoffs (and PME mesh dimensions).
            Smaller values give higher accuracy at the cost of more
            computation. Default is 1e-6, matching the nvalchemiops default.

            The Ewald cutoffs follow the Kolafa-Perram formula:
            - eta = (V^2 / N)^(1/6) / sqrt(2*pi)
            - cutoff_real = sqrt(-2 * ln(accuracy)) * eta
            - cutoff_recip = sqrt(-2 * ln(accuracy)) / eta

        Notes
        -----
        For new-format models with external Coulomb, this updates the external module.
        For legacy models with embedded Coulomb, a warning is issued as those modules
        cannot be modified at runtime.

        ``"ewald"`` and ``"pme"`` both require periodic systems (``cell`` set);
        invoking the calculator without a cell raises ``ValueError`` at
        ``prepare_input``.

        Hessian note: ``"ewald"``/``"pme"`` Hessians are computed at fixed charge
        (finite-difference of the analytic forces; the charge-response coupling
        ``d^2E/(dq.dr)`` through the model's predicted charges is omitted), while
        ``"dsf"`` Hessians are relaxed-charge (fully autograd). Vibrational
        frequencies / IR intensities are therefore not directly comparable across
        these backends.
        """
        if method not in ("simple", "dsf", "ewald", "pme"):
            raise ValueError(f"Invalid method: {method}")

        # Warn if model has embedded Coulomb (legacy models)
        if self._has_embedded_coulomb() and self.external_coulomb is None:
            warnings.warn(
                "Model has embedded Coulomb module (legacy format). "
                "set_lrcoulomb_method() only affects external Coulomb modules. "
                "For legacy models, the Coulomb method cannot be changed at runtime.",
                stacklevel=2,
            )
            return

        # Update external LRCoulomb module if present
        if self.external_coulomb is not None:
            self.external_coulomb.method = method
            if method == "dsf":
                self.external_coulomb.dsf_alpha = dsf_alpha
                self.external_coulomb.dsf_rc = cutoff
            elif method in ("ewald", "pme"):
                self.external_coulomb.ewald_accuracy = ewald_accuracy

        # Update _coulomb_cutoff based on method
        if method == "simple":
            self._coulomb_cutoff = float("inf")
        elif method == "dsf":
            self._coulomb_cutoff = cutoff
        elif method in ("ewald", "pme"):
            # Ewald/PME estimate their own real-space cutoff per call.
            self._coulomb_cutoff = None

        # Update cutoff_lr for backward compatibility
        if self._coulomb_cutoff is not None:
            self.cutoff_lr = self._coulomb_cutoff
        else:
            # Ewald/PME - use DFTD3 cutoff if available, else None
            self.cutoff_lr = self._dftd3_cutoff if self.external_dftd3 is not None else None

        self._coulomb_method = method
        # Method/cutoff changed; the memoized auto-switch DSF state is stale.
        self._auto_dsf_state = None
        self._update_lr_nblists()
        self._clear_static_nbmat_cache()

    def set_lr_cutoff(self, cutoff: float) -> None:
        """Set the unified long-range cutoff for all LR modules.

        Parameters
        ----------
        cutoff : float
            Cutoff distance in Angstroms for LR neighbor lists.

        Notes
        -----
        This updates both _coulomb_cutoff and _dftd3_cutoff.
        Ewald/PME use their own per-call neighbor lists and ignore this cutoff.
        """
        # Update both cutoffs (but not for ewald/pme which manage their own)
        if self._coulomb_method not in ("ewald", "pme"):
            self._coulomb_cutoff = cutoff
        self._dftd3_cutoff = cutoff
        self.cutoff_lr = cutoff
        # Cutoffs changed; the memoized auto-switch DSF state is stale.
        self._auto_dsf_state = None
        self._update_lr_nblists()
        self._clear_static_nbmat_cache()

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
        # Cutoffs changed; the memoized auto-switch DSF state is stale.
        self._auto_dsf_state = None
        self._update_lr_nblists()
        self._clear_static_nbmat_cache()

    def _validate_species_and_charge(self, data: dict[str, Any]) -> None:
        """Validate input elements and net charge against model metadata.

        Raises ``ValueError`` for atomic numbers outside the model's
        ``implemented_species`` or for net-charged systems when the model
        declares ``supports_charged_systems == False``. Silent no-op for models
        that did not declare ``implemented_species`` (older ``.pt``, raw
        ``nn.Module``). Shared by :meth:`eval` and
        :meth:`hessian_vector_product` so both enforce the same contract.

        The species part costs a D2H tolist() per call, so it is skipped when
        the same ``numbers`` tensor was already validated (identity + ``_version``
        cache); the charge part inspects per-call values and always runs.
        """
        if "numbers" not in data:
            # Guarded so prepare_input still owns the missing-key error path with
            # its descriptive "Missing key numbers" message.
            return
        impl = (self.metadata or {}).get("implemented_species") or []
        if impl:
            numbers = data["numbers"]
            cache_key = self._numbers_validation_key(numbers)
            cached = self._species_validation_cache
            cache_hit = (
                cache_key is not None
                and cached is not None
                and cached[0] == cache_key
                and cached[1]() is numbers
                and cached[2] is impl
            )
            if not cache_hit:
                self._validate_numbers(numbers, impl)
                if cache_key is not None:
                    # The weakref guards a recycled id()/data_ptr() at the same
                    # address; _version in the key guards in-place mutation of a
                    # kept-alive buffer. impl is compared by identity to catch
                    # metadata swaps between calls.
                    self._species_validation_cache = (cache_key, weakref.ref(numbers), impl)
        meta = self.metadata or {}
        if meta.get("supports_charged_systems") is False:
            # torch.as_tensor handles scalars, lists, ndarrays, 0-d and N-d tensors
            # uniformly so per-system charges in batched inputs (e.g. batched-NEB)
            # don't raise the misleading "only one element tensors..." from float().
            charge_t = torch.as_tensor(data.get("charge", 0.0))
            if charge_t.numel() > 0 and float(charge_t.abs().max().item()) > 1e-6:
                bad = charge_t[charge_t.abs() > 1e-6].flatten().tolist()
                raise ValueError(
                    f"This model does not support net-charged systems "
                    f"(got non-zero charge(s) {bad}). Net-neutral zwitterions are supported. "
                    f"For ions use `isayevlab/aimnet2-wb97m-d3`. "
                    f"Pass validate_species=False to bypass."
                )

    def _validate_numbers(self, numbers: Any, impl: Any) -> None:
        """Raise for atomic numbers outside ``implemented_species`` (costs a D2H sync)."""
        # numbers may be a raw list, ndarray, or tensor — normalize first.
        seen = {int(z) for z in torch.as_tensor(numbers).flatten().tolist() if int(z) > 0}
        unsupported = sorted(seen - set(impl))
        if unsupported:
            raise ValueError(
                f"Atomic numbers {unsupported} are not in this model's "
                f"implemented_species {sorted(impl)}. This model was trained on "
                f"a restricted element set; passing other elements yields undefined "
                f"output. For broader element coverage on equilibrium structures use "
                f"`isayevlab/aimnet2-wb97m-d3`; for radicals/open-shell systems use "
                f"`isayevlab/aimnet2-nse`. Pass validate_species=False to bypass."
            )

    @staticmethod
    def _numbers_validation_key(value: Any) -> tuple[Any, ...] | None:
        """Identity key for skipping repeated species validation of one tensor.

        Unlike :meth:`_tensor_identity_key` (CUDA-only, static nbmat cache),
        this accepts any dense tensor — CPU inputs also pay the per-step
        tolist() + set-diff otherwise. ``_version`` in the key guards in-place
        mutation of a reused buffer.
        """
        if not isinstance(value, Tensor) or value.is_sparse:
            return None
        try:
            version = int(value._version)
        except RuntimeError:
            return None
        return (
            id(value),
            str(value.device),
            str(value.dtype),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            int(value.data_ptr()),
            int(value.storage_offset()),
            version,
        )

    def eval(
        self, data: dict[str, Any], forces=False, stress=False, hessian=False, *, validate_species: bool = True
    ) -> dict[str, Any]:
        """Run the model on ``data`` and return the output dict.

        For a single structure each value is a ``Tensor``. For batched Hessian
        requests the output is collected per structure: a 3D ``coord`` batch
        (B, N, 3) yields values with a new leading batch dim (stacked), while a
        multi-molecule ``mol_idx`` input yields a per-molecule ``list[Tensor]``
        for each key (molecules are independent and generally ragged).
        """
        # Species validation — opt-out via validate_species=False.
        if validate_species:
            self._validate_species_and_charge(data)
        # Warn once if the caller requests an open-shell `mult` this model ignores.
        self._maybe_warn_mult_ignored(data)
        # Hessian + torch.compile is known to hang on the double-backward
        # path through GELU activations. Fail fast instead.
        if hessian and getattr(self, "_was_compiled", False):
            raise RuntimeError(
                "Hessian computation is incompatible with compile_model=True "
                "(Dynamo + double-backward through GELU hangs). Reconstruct calculator "
                "with compile_model=False."
            )

        if hessian:
            subsystems = self._split_hessian_batch(data)
            if subsystems is not None:
                stack = torch.as_tensor(data["coord"]).ndim == 3
                return self._eval_hessian_batched(
                    subsystems, forces=forces, stress=stress, validate_species=validate_species, stack=stack
                )

        # The simple->dsf PBC auto-switch in prepare_input is scoped to this
        # evaluation: any pending restore is consumed in the finally block, so
        # an exception mid-eval cannot leave the calculator on the switched method.
        self._pbc_coulomb_restore = None
        try:
            data = self.prepare_input(data, hessian=hessian)

            if hessian and "mol_idx" in data and data["mol_idx"][-1] > 0:
                raise NotImplementedError(
                    "In-call Hessian for hand-flattened multi-molecule input is not supported; "
                    "pass a 3D batch or a mol_idx dict to get per-structure Hessians."
                )
            data = self.set_grad_tensors(data, forces=forces, stress=stress, hessian=hessian)
            if isinstance(self.model, torch.jit.ScriptModule):
                with torch.jit.optimized_execution(False):  # type: ignore
                    data = self.model(data)
            else:
                data = self.model(data)
            # Run external modules if present
            data, coulomb_terms = self._run_external_modules(
                data, forces=forces or hessian, stress=stress, hessian=hessian
            )
            data = self.get_derivatives(
                data, forces=forces, stress=stress, hessian=hessian, coulomb_terms=coulomb_terms
            )
            data = self.process_output(data)
            return data
        finally:
            restore = self._pbc_coulomb_restore
            if restore is not None:
                self._pbc_coulomb_restore = None
                # Keep the DSF-side state (incl. warmed-up adaptive neighbor
                # lists) for the next periodic call before flipping back to the
                # pre-switch method.
                self._auto_dsf_state = self._snapshot_coulomb_state()
                self._restore_coulomb_state(restore)

    def _run_external_modules(
        self,
        data: dict[str, Tensor],
        *,
        forces: bool = False,
        stress: bool = False,
        hessian: bool = False,
    ) -> tuple[dict[str, Tensor], ExternalDerivativeTerms | None]:
        """Run external Coulomb and DFTD3 modules if attached.

        External backends return ``(data, terms)`` when explicit force/virial
        derivatives are requested. Ewald/PME switch to their local training
        wrapper for force/stress training.
        """
        coulomb_terms = None
        if self.external_coulomb is not None:
            training_derivatives = (
                self.external_coulomb.method in ("ewald", "pme", "dsf")
                and getattr(self, "_train", False)
                and (forces or stress)
            )
            kwargs: dict[str, Any] = {
                "compute_forces": forces,
                "compute_virial": stress,
                "training_derivatives": training_derivatives,
                "hessian": hessian,
            }
            if training_derivatives and stress and self.external_coulomb.method in ("ewald", "pme"):
                strain_inputs = getattr(self, "_external_strain_inputs", None)
                if strain_inputs is not None:
                    kwargs.update(strain_inputs)
            result = self.external_coulomb(data, **kwargs)
            if forces or stress:
                data, coulomb_terms = result
            else:
                data = result

        dftd3_terms = None
        if self.external_dftd3 is not None:
            coord = data.get("coord")
            dftd3_energy_graph = hessian or (
                not forces and not stress and isinstance(coord, Tensor) and coord.requires_grad
            )
            if dftd3_energy_graph:
                data = self.external_dftd3(data, hessian=True)
            else:
                cache_key = self._static_dftd3_cache_key(
                    data,
                    forces=forces,
                    stress=stress,
                    hessian=hessian,
                    dftd3_energy_graph=dftd3_energy_graph,
                )
                cached = self._get_static_dftd3_cache(cache_key) if cache_key is not None else None
                if cached is not None:
                    data, dftd3_terms = self._apply_static_dftd3_cache(data, cached, forces=forces)
                else:
                    data_before_dftd3 = dict(data)
                    result = self.external_dftd3(
                        data,
                        compute_forces=forces,
                        compute_virial=stress,
                    )
                    if forces or stress:
                        data, dftd3_terms = result
                    else:
                        data = result
                    if cache_key is not None:
                        self._remember_static_dftd3(cache_key, data_before_dftd3, data, dftd3_terms)

        return data, _combine_external_terms(coulomb_terms, dftd3_terms)

    def prepare_input(self, data: dict[str, Any], *, hessian: bool = False) -> dict[str, Tensor]:
        raw_data = data
        self._static_input_cache_key = None
        self._static_input_cache_refs = None
        caller_had_mol_idx = raw_data.get("mol_idx") is not None
        caller_had_nbmat = raw_data.get("nbmat") is not None
        data = self.to_input_tensors(data)
        data = self.mol_flatten(data, hessian=hessian)
        if data.get("cell") is not None and self._coulomb_method == "simple":
            warnings.warn(
                "Switching to DSF Coulomb for PBC for this evaluation; "
                "call set_lrcoulomb_method() to select a periodic method persistently.",
                stacklevel=1,
            )
            # Scope the auto-switch to the current evaluation: eval() restores
            # this snapshot in its finally block, so one periodic call does not
            # permanently flip a gas-phase calculator off its trained "simple"
            # full Coulomb. Explicit set_lrcoulomb_method() calls stay persistent.
            prev_coulomb_state = self._snapshot_coulomb_state()
            if self._auto_dsf_state is not None:
                # Reuse the DSF-side state (incl. warmed-up adaptive neighbor
                # lists) from the previous auto-switch instead of rebuilding it
                # on every periodic step.
                self._restore_coulomb_state(self._auto_dsf_state)
            else:
                self.set_lrcoulomb_method("dsf")
            self._pbc_coulomb_restore = prev_coulomb_state
        if self._coulomb_method in ("ewald", "pme") and data.get("cell") is None:
            raise ValueError(
                f"Coulomb method '{self._coulomb_method}' requires a periodic 'cell' "
                "in the input data. Provide a (3,3) or (B,3,3) cell tensor, or switch "
                "to a non-periodic method via set_lrcoulomb_method('simple' | 'dsf')."
            )
        if data["coord"].ndim == 2:
            # Skip neighbor list calculation if already provided
            if "nbmat" not in data:
                cache_key = self._static_nbmat_cache_key(
                    raw_data,
                    data,
                    hessian=hessian,
                    caller_had_mol_idx=caller_had_mol_idx,
                    caller_had_nbmat=caller_had_nbmat,
                )
                self._static_input_cache_key = cache_key
                self._static_input_cache_refs = self._static_tensor_refs(raw_data) if cache_key is not None else None
                cached = self._get_static_nbmat_cache(cache_key, raw_data) if cache_key is not None else None
                if cached is not None and "nbmat" in cached:
                    data.update(cached)
                else:
                    data = self.make_nbmat(data)
                    if cache_key is not None:
                        self._remember_static_nbmat(cache_key, raw_data, data)
            data = self.pad_input(data)
        return data

    def _static_nbmat_cache_key(
        self,
        raw_data: dict[str, Any],
        data: dict[str, Tensor],
        *,
        hessian: bool,
        caller_had_mol_idx: bool,
        caller_had_nbmat: bool,
    ) -> tuple[Any, ...] | None:
        """Return a cache key for exact static CUDA neighbor-list reuse."""
        if (
            not self.cache_static
            or hessian
            or caller_had_mol_idx
            or caller_had_nbmat
            or self._batch is not None
            or data.get("cell") is not None
            or data["coord"].ndim != 2
            or torch.device(self.device).type != "cuda"
        ):
            return None

        raw_coord = raw_data.get("coord")
        raw_numbers = raw_data.get("numbers")
        coord_key = self._tensor_identity_key(raw_coord)
        numbers_key = self._tensor_identity_key(raw_numbers)
        if coord_key is None or numbers_key is None:
            return None

        return (
            "static-nbmat-v1",
            coord_key,
            numbers_key,
            int(data["coord"].shape[0]),
            float(self.cutoff),
            None if self.cutoff_lr is None else float(self.cutoff_lr),
            self._coulomb_method,
            None if self._coulomb_cutoff is None else float(self._coulomb_cutoff),
            float(self._dftd3_cutoff),
            int(self._max_mol_size),
            self.external_coulomb is not None,
            self.external_dftd3 is not None,
            self._nblist_lr is not None,
            self._nblist_coulomb is not None,
            self._nblist_dftd3 is not None,
        )

    @staticmethod
    def _tensor_identity_key(value: Any) -> tuple[Any, ...] | None:
        if not isinstance(value, Tensor) or value.device.type != "cuda" or value.is_sparse:
            return None
        try:
            version = int(value._version)
        except RuntimeError:
            return None
        return (
            id(value),
            str(value.device),
            str(value.dtype),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            int(value.data_ptr()),
            int(value.storage_offset()),
            version,
        )

    @staticmethod
    def _static_tensor_refs(raw_data: dict[str, Any]) -> tuple[Any, Any] | None:
        raw_coord = raw_data.get("coord")
        raw_numbers = raw_data.get("numbers")
        if not isinstance(raw_coord, Tensor) or not isinstance(raw_numbers, Tensor):
            return None
        return weakref.ref(raw_coord), weakref.ref(raw_numbers)

    def _get_static_nbmat_cache(self, cache_key: tuple[Any, ...], raw_data: dict[str, Any]) -> dict[str, Tensor] | None:
        entry = self._static_nbmat_cache.get(cache_key)
        if entry is None:
            return None
        coord_ref, numbers_ref, cached = entry
        if coord_ref() is not raw_data.get("coord") or numbers_ref() is not raw_data.get("numbers"):
            self._static_nbmat_cache.pop(cache_key, None)
            return None
        return cached

    def _remember_static_nbmat(
        self,
        cache_key: tuple[Any, ...],
        raw_data: dict[str, Any],
        data: dict[str, Tensor],
    ) -> None:
        raw_coord = raw_data.get("coord")
        raw_numbers = raw_data.get("numbers")
        if not isinstance(raw_coord, Tensor) or not isinstance(raw_numbers, Tensor):
            return
        nbmat_keys = (
            "nbmat",
            "shifts",
            "nbmat_lr",
            "shifts_lr",
            "nbmat_coulomb",
            "shifts_coulomb",
            "nbmat_dftd3",
            "shifts_dftd3",
        )
        cached = {key: data[key].detach() for key in nbmat_keys if isinstance(data.get(key), Tensor)}
        if not cached:
            return
        if len(self._static_nbmat_cache) >= 8:
            self._static_nbmat_cache.pop(next(iter(self._static_nbmat_cache)))
        self._static_nbmat_cache[cache_key] = (weakref.ref(raw_coord), weakref.ref(raw_numbers), cached)

    def _static_dftd3_cache_key(
        self,
        data: dict[str, Tensor],
        *,
        forces: bool,
        stress: bool,
        hessian: bool,
        dftd3_energy_graph: bool,
    ) -> tuple[Any, ...] | None:
        if (
            not getattr(self, "cache_static", False)
            or getattr(self, "_train", False)
            or stress
            or hessian
            or dftd3_energy_graph
            or getattr(self, "_static_input_cache_key", None) is None
            or getattr(self, "_static_input_cache_refs", None) is None
            or self.external_dftd3 is None
            or data.get("cell") is not None
        ):
            return None
        return (
            "static-dftd3-v1",
            self._static_input_cache_key,
            bool(forces),
            str(data["coord"].dtype),
            str(data["coord"].device),
            float(self.external_dftd3.s6),
            float(self.external_dftd3.s8),
            float(self.external_dftd3.a1),
            float(self.external_dftd3.a2),
            float(self.external_dftd3.smoothing_on),
            float(self.external_dftd3.smoothing_off),
            self.external_dftd3.key_out,
        )

    def _get_static_dftd3_cache(self, cache_key: tuple[Any, ...]) -> dict[str, Any] | None:
        cache = getattr(self, "_static_dftd3_cache", {})
        current_refs = getattr(self, "_static_input_cache_refs", None)
        entry = cache.get(cache_key)
        if entry is None or current_refs is None:
            return None
        coord_ref, numbers_ref, cached = entry
        current_coord = current_refs[0]()
        current_numbers = current_refs[1]()
        if coord_ref() is not current_coord or numbers_ref() is not current_numbers:
            cache.pop(cache_key, None)
            return None
        return cached

    def _apply_static_dftd3_cache(
        self,
        data: dict[str, Tensor],
        cached: dict[str, Any],
        *,
        forces: bool,
    ) -> tuple[dict[str, Tensor], ExternalDerivativeTerms | None]:
        assert self.external_dftd3 is not None
        key_out = self.external_dftd3.key_out
        energy_delta = cached["energy_delta"].to(device=data["coord"].device)
        if key_out in data:
            data[key_out] = data[key_out].double() + energy_delta.double()
        else:
            data[key_out] = energy_delta.double()

        terms = None
        if forces:
            force = cached.get("forces")
            terms = ExternalDerivativeTerms(forces=None if force is None else force.to(device=data["coord"].device))
        return data, terms

    def _remember_static_dftd3(
        self,
        cache_key: tuple[Any, ...],
        data_before: dict[str, Tensor],
        data_after: dict[str, Tensor],
        terms: ExternalDerivativeTerms | None,
    ) -> None:
        current_refs = getattr(self, "_static_input_cache_refs", None)
        if current_refs is None or self.external_dftd3 is None:
            return
        key_out = self.external_dftd3.key_out
        energy_after = data_after.get(key_out)
        if not isinstance(energy_after, Tensor):
            return
        energy_before = data_before.get(key_out)
        if isinstance(energy_before, Tensor):
            energy_delta = energy_after - energy_before.to(dtype=energy_after.dtype, device=energy_after.device)
        else:
            energy_delta = energy_after

        cached: dict[str, Any] = {"energy_delta": energy_delta.detach()}
        if terms is not None and isinstance(terms.forces, Tensor):
            cached["forces"] = terms.forces.detach()
        if len(self._static_dftd3_cache) >= 8:
            self._static_dftd3_cache.pop(next(iter(self._static_dftd3_cache)))
        self._static_dftd3_cache[cache_key] = (
            current_refs[0],
            current_refs[1],
            cached,
        )

    def _clear_static_nbmat_cache(self) -> None:
        self._static_nbmat_cache.clear()
        self._static_dftd3_cache.clear()

    def process_output(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if data["coord"].ndim == 2:
            data = self.unpad_output(data)
        data = self.mol_unflatten(data)
        data = self.keep_only(data)
        return data

    def _split_hessian_batch(self, data: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Return per-structure sub-inputs for a batched Hessian request, or None
        when the input is a single structure (handled by the normal path).

        Recognized batched forms:
          * 3D ``coord`` of shape (B, N, 3) with B > 1, and
          * flat 2D ``coord`` with a ``mol_idx`` spanning more than one molecule.
        """
        coord = torch.as_tensor(data["coord"])
        if coord.ndim == 3 and coord.shape[0] > 1:
            return self._split_batch_dim(data, int(coord.shape[0]))
        if coord.ndim == 2 and data.get("mol_idx") is not None:
            mol_idx = torch.as_tensor(data["mol_idx"])
            if mol_idx.numel() and int(mol_idx.max()) > 0:
                return self._split_mol_idx(data, mol_idx)
        return None

    @staticmethod
    def _select_for_structure(value: Any, index: int, n_struct: int) -> Any:
        """Pick the per-structure slice when ``value`` is batched along structures,
        else return it unchanged (shared across structures).

        Note: the ``shape[0] == n_struct`` test is a heuristic valid only for
        per-structure scalar quantities (charge/mult), not general per-atom tensors.
        """
        t = torch.as_tensor(value)
        return t[index] if t.ndim >= 1 and t.shape[0] == n_struct else t

    def _split_batch_dim(self, data: dict[str, Any], B: int) -> list[dict[str, Any]]:
        """Slice a (B, ...) batched input into B single-structure dicts."""
        subs: list[dict[str, Any]] = []
        for b in range(B):
            sub: dict[str, Any] = {}
            for k, v in data.items():
                if v is None:
                    continue
                if k in ("coord", "numbers"):
                    sub[k] = torch.as_tensor(v)[b]
                elif k in ("charge", "mult"):
                    sub[k] = self._select_for_structure(v, b, B)
                elif k == "cell":
                    t = torch.as_tensor(v)
                    sub[k] = t[b] if t.ndim == 3 else t
                elif k.startswith(("nbmat", "shifts")) or k == "mol_idx":
                    # Precomputed neighbor-list keys must not be shared unsliced
                    # across subsystems (they'd be wrong per-structure); the
                    # recursive eval rebuilds them.
                    continue
                else:
                    sub[k] = v
            subs.append(sub)
        return subs

    def _split_mol_idx(self, data: dict[str, Any], mol_idx: Tensor) -> list[dict[str, Any]]:
        """Split a flat, ``mol_idx``-tagged input into one dict per molecule."""
        coord = torch.as_tensor(data["coord"])
        numbers = torch.as_tensor(data["numbers"])
        cell = torch.as_tensor(data["cell"]) if data.get("cell") is not None else None
        num_molecules = int(mol_idx.max()) + 1
        subs: list[dict[str, Any]] = []
        for m in range(num_molecules):
            sel = mol_idx == m
            sub: dict[str, Any] = {"coord": coord[sel], "numbers": numbers[sel]}
            if data.get("charge") is not None:
                sub["charge"] = self._select_for_structure(data["charge"], m, num_molecules)
            if data.get("mult") is not None:
                sub["mult"] = self._select_for_structure(data["mult"], m, num_molecules)
            if cell is not None:
                sub["cell"] = cell[m] if cell.ndim == 3 else cell
            if data.get("pbc") is not None:
                sub["pbc"] = data["pbc"]
            subs.append(sub)
        return subs

    def _snapshot_coulomb_state(self) -> dict[str, Any]:
        """Snapshot the Coulomb-method state mutated by :meth:`set_lrcoulomb_method`.

        Used to scope the automatic simple->dsf PBC switch to a single
        evaluation. Neighbor-list objects are captured by reference (not
        copied), so swapping states back and forth preserves their adaptive
        ``max_neighbors`` warm-up on both sides.
        """
        attrs = (
            "_coulomb_method",
            "_coulomb_cutoff",
            "cutoff_lr",
            "_nblist_lr",
            "_nblist_dftd3",
            "_nblist_coulomb",
        )
        state: dict[str, Any] = {name: getattr(self, name, _SENTINEL) for name in attrs}
        if self.external_coulomb is not None:
            state["_external_coulomb_attrs"] = {
                name: getattr(self.external_coulomb, name, _SENTINEL)
                for name in ("method", "dsf_alpha", "dsf_rc", "ewald_accuracy")
            }
        else:
            state["_external_coulomb_attrs"] = _SENTINEL
        return state

    def _restore_coulomb_state(self, state: dict[str, Any]) -> None:
        """Apply state captured by :meth:`_snapshot_coulomb_state`.

        Non-destructive on ``state`` — the memoized DSF-side snapshot is
        re-applied on every periodic call.
        """
        external_attrs = state.get("_external_coulomb_attrs", _SENTINEL)
        for name, val in state.items():
            if name == "_external_coulomb_attrs":
                continue
            if val is _SENTINEL:
                if hasattr(self, name):
                    delattr(self, name)
            else:
                setattr(self, name, val)
        if external_attrs is not _SENTINEL and self.external_coulomb is not None:
            for name, val in external_attrs.items():
                if val is _SENTINEL:
                    if hasattr(self.external_coulomb, name):
                        delattr(self.external_coulomb, name)
                else:
                    setattr(self.external_coulomb, name, val)

    def _snapshot_eval_state(self) -> dict[str, Any]:
        """Snapshot mutable calculator state touched by nested eval-style calls."""
        shallow_attrs = (
            "_batch",
            "_max_mol_size",
            "_static_input_cache_key",
            "_static_input_cache_refs",
            "_saved_for_grad",
            "_coulomb_method",
            "_coulomb_cutoff",
            "cutoff_lr",
            "_external_strain_inputs",
        )
        copied_attrs = ("_nblist_lr", "_nblist_dftd3", "_nblist_coulomb")
        state = {name: getattr(self, name, _SENTINEL) for name in shallow_attrs}
        for name in copied_attrs:
            val = getattr(self, name, _SENTINEL)
            state[name] = _SENTINEL if val is _SENTINEL else copy.deepcopy(val)
        if self.external_coulomb is not None:
            state["_external_coulomb_attrs"] = {
                name: getattr(self.external_coulomb, name, _SENTINEL)
                for name in ("method", "dsf_alpha", "dsf_rc", "ewald_accuracy")
            }
        else:
            state["_external_coulomb_attrs"] = _SENTINEL
        return state

    def _restore_eval_state(self, state: dict[str, Any]) -> None:
        """Restore state captured by :meth:`_snapshot_eval_state`."""
        external_attrs = state.pop("_external_coulomb_attrs", _SENTINEL)
        for name, val in state.items():
            if val is _SENTINEL:
                if hasattr(self, name):
                    delattr(self, name)
            else:
                setattr(self, name, val)
        if external_attrs is not _SENTINEL and self.external_coulomb is not None:
            for name, val in external_attrs.items():
                if val is _SENTINEL:
                    if hasattr(self.external_coulomb, name):
                        delattr(self.external_coulomb, name)
                else:
                    setattr(self.external_coulomb, name, val)

    def _eval_hessian_batched(
        self,
        subsystems: list[dict[str, Any]],
        *,
        forces: bool,
        stress: bool,
        validate_species: bool,
        stack: bool = True,
    ) -> dict[str, Any]:
        """Run the single-structure Hessian path on each subsystem and collect.

        When ``stack`` is True and every subsystem yields the same shape for a
        key, that key is stacked along a new leading batch dim; otherwise (ragged
        or ``stack=False``) the per-subsystem values are returned as a list.
        Multi-molecule (``mol_idx``) inputs always use ``stack=False`` because the
        molecules are independent and generally ragged.
        """
        # Recursive eval(...) mutates instance scratch state (_batch, cache
        # keys, _saved_for_grad, ...). Each inner eval scopes the simple->dsf
        # PBC auto-switch itself; snapshot and restore the rest so a batched
        # call leaves the calculator unmodified.
        eval_state = self._snapshot_eval_state()
        try:
            results = [
                self.eval(sub, forces=forces, stress=stress, hessian=True, validate_species=validate_species)
                for sub in subsystems
            ]
        finally:
            self._restore_eval_state(eval_state)
        out: dict[str, Any] = {}
        for k in results[0]:
            vals = [r[k] for r in results]
            if stack and all(isinstance(v, Tensor) for v in vals) and all(v.shape == vals[0].shape for v in vals):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out

    def to_input_tensors(self, data: dict[str, Any]) -> dict[str, Tensor]:
        ret = {}
        for k in self.keys_in:
            if k not in data:
                raise KeyError(f"Missing key {k} in the input data")
            t = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k])
            # Preserve autograd graph when caller sets requires_grad=True (e.g. for external Hessian computation).
            # Otherwise detach to prevent unintended gradient accumulation in optimization loops.
            if not (isinstance(data[k], Tensor) and data[k].requires_grad):
                t = t.detach()
            ret[k] = t
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                t = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in_optional[k])
                if not (isinstance(data[k], Tensor) and data[k].requires_grad):
                    t = t.detach()
                ret[k] = t
        # Ensure all tensors have at least 1D shape for consistent batch processing
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)
        return ret

    def mol_flatten(self, data: dict[str, Tensor], *, hessian: bool = False) -> dict[str, Tensor]:
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
            if hessian and B != 1:
                raise NotImplementedError("Hessian calculation is not supported for batched inputs with B > 1")
            # Force flattening for PBC (cell present) to ensure make_nbmat computes proper neighbor lists with shifts
            if (
                hessian
                or self.nb_threshold < N
                or torch.device(self.device).type == "cpu"
                or data.get("cell") is not None
            ):
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

        cell = data.get("cell")
        if cell is not None:
            data["coord"] = move_coord_to_cell(data["coord"], cell, mol_idx, data.get("pbc"))

        N = data["coord"].shape[0]
        batch_idx = mol_idx.to(torch.int32) if mol_idx is not None else None

        # Prepare cell and pbc tensors for nvalchemiops
        if cell is not None:
            pbc = normalize_pbc(data.get("pbc"), cell, self.device)
            if cell.ndim == 2:
                cell_batched = cell.unsqueeze(0)  # (1, 3, 3)
            else:
                cell_batched = cell  # (num_systems, 3, 3)
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

        # Per-call dense Coulomb neighbor list for Ewald / PME: estimate the
        # batch-max real-space cutoff from accuracy + system geometry, build a
        # one-shot AdaptiveNeighborList for that cutoff, and alias it into
        # both `nbmat_coulomb` and `nbmat_lr` so embedded modules that fall
        # back to the shared LR list continue to work.
        if self.external_coulomb is not None and self._coulomb_method in ("ewald", "pme") and cell is not None:
            from nvalchemiops.torch.interactions.electrostatics import (
                estimate_ewald_parameters,
                estimate_pme_parameters,
            )

            accuracy = float(self.external_coulomb.ewald_accuracy)
            with torch.no_grad():
                if self._coulomb_method == "ewald":
                    params = estimate_ewald_parameters(
                        positions=data["coord"],
                        cell=cell_batched,
                        batch_idx=batch_idx,
                        accuracy=accuracy,
                    )
                else:
                    params = estimate_pme_parameters(
                        positions=data["coord"],
                        cell=cell_batched,
                        batch_idx=batch_idx,
                        accuracy=accuracy,
                    )
                rs_cut = float(params.real_space_cutoff.max().item())

            # Per-call neighbor list, sized once per eval (do not cache).
            nblist_coulomb = AdaptiveNeighborList(cutoff=rs_cut)
            nbmat_coulomb, _, shifts_coulomb = nblist_coulomb(
                positions=data["coord"],
                cell=cell_batched,
                pbc=pbc,
                batch_idx=batch_idx,
                fill_value=N,
            )
            nbmat_coulomb, shifts_coulomb = _add_padding_row(nbmat_coulomb, shifts_coulomb, N)
            data["nbmat_coulomb"] = nbmat_coulomb
            data["nbmat_lr"] = nbmat_coulomb
            assert shifts_coulomb is not None
            data["shifts_coulomb"] = shifts_coulomb
            data["shifts_lr"] = shifts_coulomb

            # DFTD3 keeps its own neighbor list flow when present.
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
                if shifts_dftd3 is not None:
                    data["shifts_dftd3"] = shifts_dftd3
            elif self._nblist_lr is not None:
                # Shared LR path active for DFTD3 only (since coulomb_cutoff is None).
                nbmat_dftd3, _, shifts_dftd3 = self._nblist_lr(
                    positions=data["coord"],
                    cell=cell_batched,
                    pbc=pbc,
                    batch_idx=batch_idx,
                    fill_value=N,
                )
                nbmat_dftd3, shifts_dftd3 = _add_padding_row(nbmat_dftd3, shifts_dftd3, N)
                data["nbmat_dftd3"] = nbmat_dftd3
                if shifts_dftd3 is not None:
                    data["shifts_dftd3"] = shifts_dftd3
            return data

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
        self._external_strain_inputs = None
        if forces or hessian:
            data["coord"].requires_grad_(True)
            self._saved_for_grad["coord"] = data["coord"]
        if stress:
            assert "cell" in data and data["cell"] is not None, "Stress calculation requires cell"
            coord_unstrained = data["coord"]
            cell = data["cell"]
            cell_unstrained = cell
            if cell.ndim == 2:
                # Single system: (3, 3) scaling
                scaling = torch.eye(3, requires_grad=True, dtype=cell.dtype, device=cell.device)
                data["coord"] = data["coord"] @ scaling
                data["cell"] = cell @ scaling
            else:
                # Batched systems: (B, 3, 3) scaling - each system gets independent scaling
                B = cell.shape[0]
                scaling = torch.eye(3, dtype=cell.dtype, device=cell.device).unsqueeze(0).expand(B, -1, -1)
                scaling.requires_grad_(True)
                mol_idx = data["mol_idx"]
                # Apply per-atom scaling: coord[i] @ scaling[mol_idx[i]]
                atom_scaling = torch.index_select(scaling, 0, mol_idx)  # (N_total, 3, 3)
                data["coord"] = (data["coord"].unsqueeze(1) @ atom_scaling).squeeze(1)
                data["cell"] = cell @ scaling
            self._saved_for_grad["scaling"] = scaling
            self._external_strain_inputs = {
                "coord_unstrained": coord_unstrained,
                "cell_unstrained": cell_unstrained,
                "scaling": scaling,
            }
        return data

    def keep_only(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        ret = {}
        for k, v in data.items():
            if k in self.keys_out or (k.endswith("_std") and k[:-4] in self.keys_out):
                ret[k] = v
        return ret

    def get_derivatives(
        self,
        data: dict[str, Tensor],
        forces: bool = False,
        stress: bool = False,
        hessian: bool = False,
        coulomb_terms: ExternalDerivativeTerms | None = None,
    ) -> dict[str, Tensor]:
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
                force = -deriv[0]
                if coulomb_terms is not None and coulomb_terms.forces is not None:
                    force = force + coulomb_terms.forces.to(dtype=force.dtype, device=force.device)
                data["forces"] = force
            if stress:
                dedc = deriv[0] if not forces else deriv[1]
                if coulomb_terms is not None and coulomb_terms.virial is not None:
                    virial = coulomb_terms.virial.to(dtype=dedc.dtype, device=dedc.device)
                    if dedc.ndim == 2 and virial.ndim == 3:
                        virial = virial.sum(dim=0)
                    # nvalchemiops virial convention is W = -dE/dstrain.
                    # AIMNet applies row-vector strain as coord @ scaling,
                    # so the stress numerator contribution is -W.T.
                    dedc = dedc - virial.mT
                cell = data["cell"].detach()
                if cell.ndim == 2:
                    volume = cell.det().abs()
                else:
                    volume = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                data["stress"] = dedc / volume
        if hessian:
            H = self.calculate_hessian(data["forces"], self._saved_for_grad["coord"])
            if coulomb_terms is not None and getattr(coulomb_terms, "hessian", None) is not None:
                # The LR coulomb hessian is computed in float64 via finite
                # differences. Accumulate in that (higher) precision rather than
                # downcasting it to H's dtype, which would discard the FD precision.
                H = H.to(dtype=coulomb_terms.hessian.dtype, device=H.device) + coulomb_terms.hessian.to(device=H.device)
            data["hessian"] = H
        return data

    @staticmethod
    def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
        """Dense ``(N, 3, N, 3)`` Hessian of the energy w.r.t. real-atom coordinates.

        Autograd contract (IMPORTANT):
        The returned dense Hessian is a **detached value**: it carries no
        autograd graph back to the coordinates or model parameters. This is by
        design (it is materialized via ``torch.func.vmap`` over a vjp of the
        already-built force graph, and the periodic Ewald/PME block is a
        fixed-charge finite-difference term that is non-differentiable). Forces
        DO compose with an upstream coordinate-builder graph, but the Hessian
        does not, so you cannot backpropagate through ``eval(..., hessian=True)``.

        If you need the Hessian to *compose* (e.g. ``H @ v`` that scales with /
        differentiates through an outer computation) or to avoid forming the
        dense ``(N, 3, N, 3)`` tensor on large systems, use the matrix-free
        :meth:`hessian_vector_product` instead. For a fully-differentiable
        Hessian, build one externally with
        ``torch.autograd.functional.hessian(energy_fn, coords)`` over a closure
        that calls the model on differentiable coordinates (note that the
        periodic Ewald/PME long-range block remains a fixed-charge FD term in
        either case).
        """
        # Coord includes padding atom (shape N+1), forces only for real atoms (shape N).
        # Hessian computed only for actual atoms: (N, 3, N, 3).
        #
        # vmap-over-vjp form (not is_grads_batched=True or autograd.functional.hessian):
        # torch.library.register_vmap on aimnet::conv_sv_2d_sp_{bwd,bwd_bwd} is consulted
        # ONLY by the functorch dispatch (torch.func.vmap). The legacy batching dispatch
        # would still raise "Batching rule not implemented."
        n = forces.numel()
        eye = torch.eye(n, device=forces.device, dtype=forces.dtype)

        def vjp(go: Tensor) -> Tensor:
            return torch.autograd.grad(
                forces.flatten(),
                coord,
                grad_outputs=go,
                retain_graph=True,
                allow_unused=True,
            )[0]

        hessian = -torch.func.vmap(vjp, 0)(eye)
        return hessian.view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]

    @torch.inference_mode(False)
    @torch.enable_grad()
    def hessian_vector_product(
        self,
        data: dict[str, Any],
        vectors: Tensor,
        *,
        eps: float = 5e-4,
        validate_species: bool = True,
        create_graph: bool = False,
    ) -> Tensor:
        """Matrix-free Hessian-vector product(s) ``H @ v`` for one structure.

        Computes ``H @ v`` without forming the dense ``(N, 3, N, 3)`` Hessian,
        enabling Lanczos/LOBPCG negative-curvature checks and CG-Newton
        preconditioning on large systems.

        Parameters
        ----------
        data : dict
            Single-structure input (same keys as ``eval``). 3D batched or
            multi-molecule ``mol_idx`` inputs are not supported.
        vectors : Tensor
            Direction(s), shape ``(N, 3)`` or ``(K, N, 3)`` over the real atoms.
        eps : float
            Central-difference step (Angstrom) for the periodic Ewald/PME
            long-range term. Ignored for ``simple``/``dsf``.
        create_graph : bool
            If ``True``, keep the differentiable autograd block of the HVP in
            the graph so it can compose with an outer loss. The Ewald/PME
            fixed-position finite-difference block remains detached. Default
            ``False`` preserves the numeric/detached operator-action behavior.

        Returns
        -------
        Tensor
            ``H @ v``, shape ``(N, 3)`` or ``(K, N, 3)``, matching ``vectors``.

        Notes
        -----
        The autograd part (NN + short-range + ``simple``/``dsf`` Coulomb +
        DFTD3) is an exact reverse-mode product. For ``ewald``/``pme`` the
        long-range block is a fixed-charge directional finite difference (2
        force evals per vector); the same charge-response and step caveats as
        the dense Ewald/PME Hessian apply (see
        :meth:`aimnet.modules.lr.LRCoulomb._coul_nvalchemi_fd_hessian`). This
        mirrors the dense :meth:`calculate_hessian` assembly term-by-term, so
        ``hessian_vector_product(v)`` equals ``H.reshape(3N, 3N) @ v`` to the
        backend's tolerance. The default return is detached; set
        ``create_graph=True`` when the differentiable autograd block must
        compose with an outer computation. See :meth:`calculate_hessian` for
        the detached-Hessian contract and the fully-differentiable recipe.

        Integration note: the external modules run via
        ``_run_external_modules(forces=False, hessian=(method == 'dsf'))`` --
        ENERGY-GRAPH mode so every external term (DFTD3 + Coulomb) stays
        differentiable w.r.t. ``coord`` and its curvature is captured by the
        autograd vjp. ``forces=False`` (not ``True``) is required: with
        ``forces=True`` the DFTD3 branch takes its detached explicit-force path
        and its second-derivative curvature is silently dropped from ``H @ v``.
        The ``hessian`` flag controls the dsf-vs-ewald/pme split: dsf passes
        ``hessian=True`` so ``LRCoulomb.forward`` routes through its
        differentiable closed-form torch path (``_coul_dsf_torch``), keeping the
        dsf curvature in the autograd graph (dsf has no dense FD block, so this is
        free); ewald/pme pass ``hessian=False`` so the dense O(2*3N) FD block is
        NOT computed, while the periodic energy is still added
        differentiable-through-charges (capturing the charge-response curvature).
        The autograd vjp of the differentiable forces equals the dense autograd
        Hessian block (NN + short-range + DFTD3 + Coulomb-charge-response), and
        the directional FD helper adds the remaining full-periodic block --
        matching the dense assembly term-by-term.

        Dtype / differentiability / eigensolver caveats:

        * Return dtype is the model dtype (typically float32) for ``simple`` and
          ``dsf``, and **float64** for ``ewald``/``pme`` (the periodic
          finite-difference block is accumulated in double precision, matching the
          dense Ewald/PME Hessian).
        * With ``create_graph=False`` the returned product is detached numeric
          operator action. With ``create_graph=True`` the autograd block remains
          differentiable w.r.t. graph-attached coordinates / model parameters;
          vectors are still treated as numeric directions, and the periodic FD
          block remains detached.
        * For ``ewald``/``pme`` the operator is symmetric only to
          finite-difference accuracy (O(eps^2)); for Lanczos/LOBPCG
          smallest/most-negative-eigenvalue (transition-state) work, pass all
          probe vectors together as a single ``(K, N, 3)`` batch so the charge
          state is frozen across the iteration, and consider symmetrizing the
          operator or tuning ``eps``.
        * The fixed-charge periodic approximation (and the ``dsf`` relaxed-charge
          vs ``ewald``/``pme`` fixed-charge asymmetry) is inherited from the dense
          Ewald/PME Hessian and can shift near-zero/negative eigenvalues for
          strongly polar periodic systems; see
          :meth:`aimnet.modules.lr.LRCoulomb._coul_nvalchemi_fd_hessian`.
        """
        if getattr(self, "_was_compiled", False):
            raise RuntimeError(
                "hessian_vector_product is incompatible with compile_model=True "
                "(Dynamo + double-backward through GELU hangs). Reconstruct with compile_model=False."
            )
        # Same species/charge validation contract as `eval` (opt-out via
        # validate_species=False); otherwise unsupported elements / charged
        # systems would yield undefined output silently.
        if validate_species:
            self._validate_species_and_charge(data)
        # Warn once if the caller requests an open-shell `mult` this model ignores.
        self._maybe_warn_mult_ignored(data)
        coord_in = torch.as_tensor(data["coord"])
        if coord_in.ndim == 3 and coord_in.shape[0] > 1:
            raise NotImplementedError("hessian_vector_product supports a single structure only (got 3D batch).")
        if coord_in.ndim == 2 and data.get("mol_idx") is not None:
            mol_idx_t = torch.as_tensor(data["mol_idx"])
            if mol_idx_t.numel() and int(mol_idx_t.max()) > 0:
                raise NotImplementedError(
                    "hessian_vector_product supports a single structure only (got mol_idx batch)."
                )

        # prepare_input / set_grad_tensors mutate instance scratch state and can
        # persistently flip Coulomb method/cutoff/list-builder state via
        # prepare_input's simple->dsf PBC auto-switch. Snapshot now and restore
        # in the finally so a single HVP call leaves the calculator unmodified.
        # The per-vector loop reads self._saved_for_grad and self.external_coulomb,
        # so the result must be fully computed inside the try before state is
        # restored.
        eval_state = self._snapshot_eval_state()
        try:
            result = self._hessian_vector_product_impl(data, vectors, eps=eps, create_graph=create_graph)
        finally:
            self._restore_eval_state(eval_state)
            # _restore_eval_state already undid any PBC auto-switch from
            # prepare_input; drop the (now-redundant) pending restore marker.
            self._pbc_coulomb_restore = None
        return result

    def _hessian_vector_product_impl(
        self, data: dict[str, Any], vectors: Tensor, *, eps: float, create_graph: bool
    ) -> Tensor:
        """Core HVP computation; instance-state snapshot/restore is handled by
        :meth:`hessian_vector_product`. See that method for the contract."""
        # Deliberate parallel forward path: this mirrors `eval` +
        # `_run_external_modules` but builds an autograd-differentiable energy
        # WITHOUT the dense periodic FD Hessian. Keep it in sync if the main
        # forward path changes.
        prepared = self.prepare_input(data, hessian=True)
        if "mol_idx" in prepared and prepared["mol_idx"][-1] > 0:
            raise NotImplementedError("hessian_vector_product supports a single structure only.")
        prepared = self.set_grad_tensors(prepared, forces=True, hessian=True)
        if isinstance(self.model, torch.jit.ScriptModule):
            with torch.jit.optimized_execution(False):  # type: ignore
                prepared = self.model(prepared)
        else:
            prepared = self.model(prepared)

        method = self._coulomb_method
        # Run the external modules in ENERGY-GRAPH mode (forces=False) so every
        # external contribution stays differentiable w.r.t. ``coord`` and its
        # second-derivative curvature is captured by the autograd vjp below. The
        # HVP does NOT use the explicit forces these modules can return -- it
        # recomputes ``forces_diff = -autograd.grad(E, coord)`` itself -- so we
        # only need the ENERGY in the graph, not the detached explicit-force path.
        #
        #   * DFTD3 (if attached -- the production default): with ``forces=False``
        #     and ``coord.requires_grad=True``, ``_run_external_modules`` takes its
        #     in-graph branch (``dftd3_energy_graph`` True), so the D3 energy is
        #     differentiable and its curvature enters ``H @ v``. This matches the
        #     dense path, which keeps D3 in-graph via ``hessian=True``. (With
        #     ``forces=True`` D3 would take its DETACHED explicit-force path and its
        #     curvature would be silently dropped from the product.)
        #   * dsf: ``hessian=True`` routes ``LRCoulomb.forward`` through its
        #     DIFFERENTIABLE closed-form torch path (``_coul_dsf_torch``), so the
        #     dsf curvature is in the autograd graph. dsf has no dense FD block, so
        #     ``hessian=True`` adds no wasteful work, and this guarantees correct
        #     routing regardless of how the (process-wide cached) model's embedded
        #     Coulomb method was last set by another caller.
        #   * ewald/pme/simple: ``hessian=False`` and ``forces=False`` so the dense
        #     O(2*3N) Ewald/PME FD block is NOT computed; the periodic energy is
        #     still added differentiable-through-charges (capturing the
        #     charge-response curvature d^2E/(dq.dr)). The full-periodic
        #     fixed-position curvature is supplied per-vector by the directional FD
        #     helper below. The Coulomb branch returns just ``data`` (no terms) for
        #     forces=False, but ``_run_external_modules`` always returns the
        #     ``(data, terms)`` 2-tuple, so the unpacking below is safe.
        # The autograd vjp therefore captures NN + short-range + DFTD3 +
        # Coulomb-charge-response, matching the dense assembly term-by-term.
        external_hessian = method == "dsf"
        prepared, _coulomb_terms = self._run_external_modules(
            prepared, forces=False, stress=False, hessian=external_hessian
        )

        coord = self._saved_for_grad["coord"]  # (N+1, 3), requires_grad
        tot_energy = prepared["energy"].sum()
        # Differentiable part of the forces only. The detached ``coulomb_terms``
        # forces are a constant w.r.t. coord (zero second derivative), so they are
        # intentionally excluded from the vjp; the periodic curvature they would
        # have carried is supplied by the directional FD helper instead.
        forces_diff = -torch.autograd.grad(tot_energy, coord, create_graph=True)[0]  # (N+1, 3)
        N = coord.shape[0] - 1

        device = coord.device
        vecs = torch.as_tensor(vectors, device=device)
        single = vecs.ndim == 2
        if single:
            vecs = vecs.unsqueeze(0)
        if vecs.shape[-2:] != (N, 3):
            raise ValueError(f"vectors must have trailing shape ({N}, 3); got {tuple(vecs.shape)}")

        outs = []
        for k in range(vecs.shape[0]):
            v = vecs[k].to(forces_diff.dtype)
            v_full = torch.zeros_like(coord)
            v_full[:N] = v
            # autograd Hv = -d(forces . v)/dcoord = d^2E/dr^2 . v
            # (NN + short-range + dsf/simple charge-response + dftd3)
            hv_full = -torch.autograd.grad(
                forces_diff.flatten(),
                coord,
                grad_outputs=v_full.flatten(),
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
            hv = hv_full[:N]
            if method in ("ewald", "pme") and self.external_coulomb is not None:
                # Full-periodic fixed-position curvature (directional FD).
                hv = hv.to(torch.float64) + self.external_coulomb._coul_nvalchemi_fd_hvp(
                    prepared, backend=method, vec=v, step=eps
                )
            outs.append(hv)
        result = torch.stack(outs, 0)
        return result[0] if single else result


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


def normalize_pbc(pbc: Tensor | None, cell: Tensor, device: str | torch.device) -> Tensor:
    """Return PBC flags as ``(B, 3)`` bool tensor matching ``cell``."""
    num_systems = 1 if cell.ndim == 2 else cell.shape[0]
    if pbc is None:
        return torch.ones((num_systems, 3), dtype=torch.bool, device=device)
    pbc = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    if pbc.ndim == 1:
        if pbc.shape[0] != 3:
            raise ValueError("pbc must have shape (3,) or (B, 3)")
        return pbc.unsqueeze(0).expand(num_systems, -1)
    if pbc.ndim == 2 and pbc.shape == (num_systems, 3):
        return pbc
    raise ValueError(f"pbc must have shape (3,) or ({num_systems}, 3), got {tuple(pbc.shape)}")


def _wrap_fractional(coord_f: Tensor, pbc: Tensor) -> Tensor:
    pbc = pbc.to(device=coord_f.device, dtype=torch.bool)
    while pbc.ndim < coord_f.ndim:
        pbc = pbc.unsqueeze(-2)
    return torch.where(pbc, coord_f % 1, coord_f)


def move_coord_to_cell(
    coord: Tensor,
    cell: Tensor,
    mol_idx: Tensor | None = None,
    pbc: Tensor | None = None,
) -> Tensor:
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
    pbc : Tensor | None
        Periodic axes, shape (3,) or (B, 3). Defaults to all periodic axes.

    Returns
    -------
    Tensor
        Coordinates wrapped into the cell.
    """
    pbc = normalize_pbc(pbc, cell, coord.device)
    if cell.ndim == 2:
        # Single cell (3, 3)
        cell_inv = torch.linalg.inv(cell)
        coord_f = coord @ cell_inv
        coord_f = _wrap_fractional(coord_f, pbc[0])
        return coord_f @ cell
    else:
        # Batched cells (B, 3, 3)
        if coord.ndim == 3:
            # Batched coords (B, N, 3) with batched cells (B, 3, 3)
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            coord_f = torch.bmm(coord, cell_inv)  # (B, N, 3)
            coord_f = _wrap_fractional(coord_f, pbc)
            return torch.bmm(coord_f, cell)
        else:
            # Flat coords (N_total, 3) with batched cells (B, 3, 3) - need mol_idx
            assert mol_idx is not None, "mol_idx required for batched cells with flat coordinates"
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            # Get cell and cell_inv for each atom
            atom_cell = cell[mol_idx]  # (N_total, 3, 3)
            atom_cell_inv = cell_inv[mol_idx]  # (N_total, 3, 3)
            atom_pbc = pbc[mol_idx]
            coord_f = torch.bmm(coord.unsqueeze(1), atom_cell_inv).squeeze(1)  # (N_total, 3)
            coord_f = _wrap_fractional(coord_f, atom_pbc)
            return torch.bmm(coord_f.unsqueeze(1), atom_cell).squeeze(1)
