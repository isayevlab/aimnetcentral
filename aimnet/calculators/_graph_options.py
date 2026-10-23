"""Resolve graph opt-in and its runtime requirements without importing the optional execution backend."""

import importlib.metadata
import os
import re
import sys

import torch

# Minimum versions for CUDA-graph capture. nvalchemi-toolkit-ops 0.5.0 orders its Warp neighbor kernels on
# PyTorch's current stream and resets buffers internally (see ``neighbor_buffers.ReusableNeighborMatrix``); graphs
# captured with 0.4.1 were measured to replay wrong energies (1-4700 eV off). 0.5.0 itself requires warp-lang >= 1.16. The package floor stays at the regular path's requirements, so older
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


# PEP 440 public version with optional local label (normalized forms and the usual spellings). ``packaging`` is not a
# core dependency, so the few fields the gate needs are parsed here.
_VERSION = re.compile(
    r"""
    ^\s*v?
    (?:(?P<epoch>\d+)!)?
    (?P<release>\d+(?:\.\d+)*)
    (?P<pre>[-_.]?(?:a|alpha|b|beta|c|rc|pre|preview)[-_.]?\d*)?
    (?P<post>-\d+|[-_.]?(?:post|rev|r)[-_.]?\d*)?
    (?P<dev>[-_.]?dev[-_.]?\d*)?
    (?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def version_at_least(text: str, minimum: tuple[int, ...]) -> bool:
    """Whether version ``text`` is at least the final release ``minimum`` under PEP 440 ordering.

    Pre-releases and development builds of ``minimum`` itself precede it and are rejected (``0.5.0rc1``,
    ``0.5.0.dev3``); post-releases and later releases, including their development builds (``0.5.1.dev0``), pass.
    Unparseable versions are rejected."""
    match = _VERSION.match(text)
    if match is None:
        return False
    if int(match.group("epoch") or 0) > 0:
        return True
    release = tuple(int(part) for part in match.group("release").split("."))
    width = max(len(release), len(minimum))
    release, padded_min = release + (0,) * (width - len(release)), minimum + (0,) * (width - len(minimum))
    if release != padded_min:
        return release > padded_min
    if match.group("pre"):
        return False
    return not (match.group("dev") and not match.group("post"))


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
        if version is None or not version_at_least(version, minimum):
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
