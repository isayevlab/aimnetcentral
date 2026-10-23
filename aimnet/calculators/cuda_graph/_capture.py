"""The stream on which the engines of this package are warmed up and captured."""

from __future__ import annotations

import threading

import torch

_STREAMS: dict[int, torch.cuda.Stream] = {}
_STREAMS_LOCK = threading.Lock()


def capture_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device, shared by every warm-up and capture (thread-safe creation).

    PyTorch keeps a cuBLAS workspace per (handle, stream) for the life of the process. Warming up on the capture
    stream allocates it there, in the regular allocator pool; a capture on a stream without a workspace allocates it
    inside the graph's pool, where it keeps a segment reserved after every graph of the pool is gone (measured: 42 MiB
    of a released pool). One shared stream also avoids a new workspace per capture, which a fresh warm-up stream per
    capture costs."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    with _STREAMS_LOCK:
        stream = _STREAMS.get(index)
        if stream is None:
            stream = _STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


def warm_up(fn, x: torch.Tensor, warmup: int) -> torch.cuda.Stream:
    """``fn(x)`` ``warmup`` times on the capture stream, which is returned for the capture.

    Ordering is stream-scoped: the capture stream waits for the caller's stream before the warm-up and the caller's
    stream waits for the warm-up after it. No device-wide synchronize or ``empty_cache`` happens here; the capture
    that follows (``torch.cuda.graph``) synchronizes the device and empties the allocator cache itself, once per
    capture (PyTorch's own behavior for every capture). Captures are rare: at most one per new layout."""
    stream = capture_stream(x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn(x)
    torch.cuda.current_stream(x.device).wait_stream(stream)
    return stream
