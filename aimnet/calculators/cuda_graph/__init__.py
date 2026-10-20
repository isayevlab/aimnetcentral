"""CUDA-graph replay of repeated ``AIMNet2Calculator`` evaluations.

``CUDAGraphCache`` is created by ``AIMNet2Calculator(..., cuda_graph=True)`` (opt-in; disabled by default) and
handles the calls it supports with static-layout engines replayed as CUDA graphs:

* ``dense``: ``N <= nb_threshold`` (dense mode), shared model computation;
* ``sparse``: ``N > nb_threshold`` (neighbor-list mode), shared model computation.

Both paths use the shared optimized primitives only when ``inference_kernels=True``;
eligible Coulomb calculations then use ``aimnet.kernels.coulomb_wp``.

Every call it does not support runs on the regular code path, unchanged.
"""

from .cache import CUDAGraphCache, cuda_graph_enabled

__all__ = ["CUDAGraphCache", "cuda_graph_enabled"]
