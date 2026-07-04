"""UMIC: Unified-Memory Inference Compiler — Alpamayo runtime.

Measurement-guided fusion/scheduling layer for multi-stage transformer
models on unified-memory edge GPUs (Jetson AGX Thor, SM 11.0).

No checkpoint modification, no quantization: weights are shared and only
the execution schedule of the same math changes. Every fusion has an
eager fallback, so a non-matching model simply runs unmodified.

Entry point:

    import umic
    report = umic.apply(model)   # patch all adopted fusions in place
"""

from umic.optimize import UmicConfig, apply

__all__ = ["apply", "UmicConfig"]
__version__ = "0.1.0"
