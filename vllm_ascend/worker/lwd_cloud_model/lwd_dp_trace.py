# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Per-rank counters for locating cross-DP collective divergence (debug only).

Set ``VLLM_LWD_DP_TRACE=1`` before starting the engine. Hot-path collectives
only increment a per-process counter; step-level events log one line carrying
every counter of this rank. When a run wedges in an EP/DP collective, compare
the last snapshot of each rank: the rank group with one extra round is the side
that diverged.
"""

import os
from collections import Counter

from vllm.logger import logger

_ENABLED = os.getenv("VLLM_LWD_DP_TRACE") == "1"
_COUNTERS: Counter[str] = Counter()


def lwd_dp_count(kind: str) -> None:
    """Count one collective without logging; safe on the MoE hot path."""
    if _ENABLED:
        _COUNTERS[kind] += 1


def lwd_dp_trace(kind: str, **fields: object) -> None:
    """Log one step-level snapshot of all counters of this process."""
    if not _ENABLED:
        return
    _COUNTERS[kind] += 1
    counters = " ".join(f"{name}={count}" for name, count in sorted(_COUNTERS.items()))
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.warning("[LWD][dp-trace] pid=%d %s %s %s", os.getpid(), kind, details, counters)
