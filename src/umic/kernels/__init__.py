"""L0: kernel registry — fused-pattern name -> best available implementation.

Every entry has an eager fallback; Triton implementations register
themselves only if Triton imports and compiles on this device (M0 gate).
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

_KERNELS: dict[str, Callable] = {}


def register_kernel(pattern_name: str, fn: Callable) -> None:
    _KERNELS[pattern_name] = fn
    logger.info("kernel registered: %s -> %s", pattern_name, fn.__name__)


def get_kernel(pattern_name: str) -> Callable | None:
    return _KERNELS.get(pattern_name)
