"""Resolve graph opt-in and its runtime requirements without importing the optional execution backend."""

import importlib.metadata
import os
import re
import sys

import torch

# Minimum versions for CUDA-graph capture. nvalchemi-toolkit-ops 0.5.0 launches its Warp neighbor kernels on
# PyTorch's current stream and orders buffer resets internally (see ``neighbor_buffers.ReusableNeighborMatrix``);
# with 0.4.1 those launches escape the capture stream and replays return wrong energies (measured 1-4700 eV off).
# 0.5.0 itself requires warp-lang >= 1.16. The package floor stays at the regular path's requirements, so older
# installs keep working and only graph capture is refused.
GRAPH_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "nvalchemi-toolkit-ops": (0, 5, 0),
    "warp-lang": (1, 16, 0),
}


def cuda_graph_enabled(cuda_graph: bool | None = False) -> bool:
    """Explicit booleans win; ``None`` opts into environment-based selection.

    An omitted argument is ``False`` even if ``AIMNET_CUDA_GRAPH`` is set.
    Environment selection enables graphs only for 1/true/yes/on, ignoring case.
    """
    if cuda_graph is not None:
        return bool(cuda_graph)
    return os.environ.get("AIMNET_CUDA_GRAPH", "0").strip().lower() in ("1", "true", "yes", "on")


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """Leading numeric release of a version string (``"1.16.0rc1"`` -> ``(1, 16, 0)``), or None."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", text)
    return None if match is None else tuple(int(part) for part in match.group(1).split("."))


def installed_versions() -> dict[str, str | None]:
    """Installed versions of the graph backend's dependencies (None when the distribution is not found)."""
    found: dict[str, str | None] = {}
    for name in GRAPH_MIN_VERSIONS:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def cuda_graph_dependency_problem() -> str | None:
    """Why the installed dependencies cannot capture correct CUDA graphs, or None when they can."""
    too_old = []
    for name, version in installed_versions().items():
        minimum = GRAPH_MIN_VERSIONS[name]
        parsed = None if version is None else _version_tuple(version)
        # Pad so that "1.16" compares equal to (1, 16, 0).
        if parsed is None or parsed + (0,) * (len(minimum) - len(parsed)) < minimum:
            required = ".".join(map(str, minimum))
            too_old.append(f"{name} {version or 'unknown'} (needs >= {required})")
    if not too_old:
        return None
    return (
        "the installed " + ", ".join(too_old) + " cannot capture correct CUDA graphs; "
        'upgrade with `pip install -U "nvalchemi-toolkit-ops>=0.5.0" "warp-lang>=1.16"`'
    )


def cuda_graph_platform_problem() -> str | None:
    """Why this PyTorch build cannot run the graph backend, or None."""
    if getattr(torch.version, "hip", None):
        return "ROCm builds are not supported (the Warp/nvalchemi kernels are CUDA-only)"
    return None


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def external_stacklevel() -> int:
    """``stacklevel`` for a ``warnings.warn`` issued by the caller of this function that attributes the warning to
    the first frame outside the ``aimnet`` package (the user's call, however deep the internal dispatch is)."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level
