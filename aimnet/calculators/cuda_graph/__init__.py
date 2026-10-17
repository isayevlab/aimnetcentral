"""CUDA-graph replay of repeated ``AIMNet2Calculator`` evaluations (PyTorch + Warp).

``CUDAGraphCache`` is created by ``AIMNet2Calculator(..., cuda_graph=None)`` (on unless ``AIMNET_CUDA_GRAPH=0``) and
handles the calls it supports with static-layout engines replayed as CUDA graphs:

* ``dense``: ``N <= nb_threshold`` (dense mode), PyTorch;
* ``sparse``: ``N > nb_threshold`` (neighbor-list mode), PyTorch and the Warp full-Coulomb kernel
  (``aimnet.kernels.coulomb_wp``).

Every call it does not support runs on the regular code path, unchanged.
"""

from .cache import CUDAGraphCache, cuda_graph_enabled

__all__ = ["CUDAGraphCache", "cuda_graph_enabled"]
