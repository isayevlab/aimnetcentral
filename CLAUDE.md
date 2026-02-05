# Claude Code Project Rules

## Project Overview

AIMNet2 is a neural network interatomic potential for molecular simulations. Key components:

- **`aimnet/calculators/`** - Main calculator (`AIMNet2Calculator`) with adaptive neighbor lists
- **`aimnet/kernels/`** - NVIDIA Warp CUDA kernels for GPU acceleration
- **`aimnet/modules/`** - Neural network modules (AEV, long-range Coulomb, DFT-D3)
- **`aimnet/models/`** - Model definitions and utilities
- **`tests/`** - Comprehensive test suite with pytest

## Code Conventions

### PyTorch Patterns

- Use `@torch.library.custom_op` for custom CUDA operations (torch.compile compatible)
- Register fake tensors with `@torch.library.register_fake` for tracing
- Prefer `torch.einsum` for batched matrix operations
- Always ensure tensor contiguity before passing to Warp kernels

### Performance Guidelines

- Avoid GPU-CPU synchronization in hot paths (no `.item()` in loops)
- Use aligned allocations (multiples of 16) for GPU buffers
- Cache dtype conversions (e.g., `idx.to(torch.int32)`) when possible
- Support both dense mode (small molecules) and sparse mode (large systems)

### Testing Requirements

- Run `make check` before committing (ruff, markdownlint, prettier)
- Run `make test` for full test suite with coverage
- Tests must cover: forward, backward, double backward, TorchScript compatibility
- Use `pytest.mark.parametrize` for testing multiple configurations

## Git Authorship Policy

**STRICT RULE**: All commits must be authored exclusively by the repository owner.

- **Author Name**: Olexandr Isayev
- **GitHub Username**: isayev
- **GitHub Email**: olexandr@olexandrisayev.com

### Requirements

1. **No AI co-authorship**: Never add `Co-Authored-By` lines for Claude or any AI assistant
2. **No co-author trailers**: Commit messages must not include any co-author attribution
3. **Single author only**: All commits must show only the repository owner as author
4. **Do not modify git config**: Never change user.name or user.email settings

### Commit Message Format

- Use clear, concise commit messages
- Do NOT append any `Co-Authored-By` trailers
- Do NOT mention AI assistance in commit messages
