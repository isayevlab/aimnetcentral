"""Resolve graph opt-in without importing the optional execution backend."""

import os


def cuda_graph_enabled(cuda_graph: bool | None = False) -> bool:
    """Explicit booleans win; ``None`` opts into environment-based selection.

    An omitted argument is ``False`` even if ``AIMNET_CUDA_GRAPH`` is set.
    Environment selection enables graphs only for 1/true/yes/on, ignoring case.
    """
    if cuda_graph is not None:
        return bool(cuda_graph)
    return os.environ.get("AIMNET_CUDA_GRAPH", "0").strip().lower() in ("1", "true", "yes", "on")
