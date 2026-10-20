"""The stream on which the engines of this package are warmed up and captured."""

from __future__ import annotations

import torch

_STREAMS: dict[int, torch.cuda.Stream] = {}


def capture_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device, shared by every warm-up and capture.

    PyTorch keeps a cuBLAS workspace per (handle, stream) for the life of the process. Warming up on the capture
    stream allocates it there, in the regular allocator pool; a capture on a stream without a workspace allocates it
    inside the graph's pool, where it keeps a segment reserved after every graph of the pool is gone (measured: 42 MiB
    of a released pool). One shared stream also avoids a new workspace per capture, which a fresh warm-up stream per
    capture costs."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        stream = _STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


def warm_up(fn, x: torch.Tensor, warmup: int) -> torch.cuda.Stream:
    """``fn(x)`` ``warmup`` times on the capture stream, which is returned for the capture.

    Blocks cached for other streams cannot serve this stream, so they are returned to the device first (as
    ``torch.cuda.graph`` does before capturing): the warm-up's memory does not stack on the regular path's cache
    (measured, shrinking 64 -> 1 batch: peak reserved 1904 MiB without this, 1508 MiB with it)."""
    stream = capture_stream(x.device)
    torch.cuda.empty_cache()
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn(x)
    torch.cuda.current_stream(x.device).wait_stream(stream)
    torch.cuda.synchronize()
    return stream
