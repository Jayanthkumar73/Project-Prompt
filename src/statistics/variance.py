"""Aggregation and variance analysis utilities.

Provides mean, std, and t-based 95% confidence intervals for metric arrays.
"""

from __future__ import annotations

from typing import Iterable, Dict
import math

import numpy as np
import scipy.stats as stats


def mean_std_ci(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = s / math.sqrt(n) if n > 1 else 0.0
    if n > 1:
        h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    else:
        h = 0.0

    return {"mean": round(m, 4), "std": round(s, 4), "ci_low": round(m - h, 4), "ci_high": round(m + h, 4)}
