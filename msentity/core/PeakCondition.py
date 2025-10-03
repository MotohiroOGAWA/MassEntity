from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .PeakSeries import PeakSeries

class PeakCondition(ABC):
    """
    Abstract base class for a condition to evaluate on PeakSeries.
    """

    @abstractmethod
    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        pass

    def __and__(self, other: PeakCondition) -> PeakCondition:
        return AndCondition(self, other)

    def __or__(self, other: PeakCondition) -> PeakCondition:
        return OrCondition(self, other)

    def __invert__(self) -> PeakCondition:
        return NotCondition(self)
    
class AndCondition(PeakCondition):
    """
    Condition that evaluates to True if both conditions are True.

    Example:
        combined = cond1 & cond2  # both conditions must be satisfied
    """
    def __init__(self, cond1: PeakCondition, cond2: PeakCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        return self.cond1.evaluate(peaks) & self.cond2.evaluate(peaks)


class OrCondition(PeakCondition):
    """
    Condition that evaluates to True if either condition is True.

    Example:
        either = cond1 | cond2  # at least one condition must be satisfied
    """
    def __init__(self, cond1: PeakCondition, cond2: PeakCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        return self.cond1.evaluate(peaks) | self.cond2.evaluate(peaks)


class NotCondition(PeakCondition):
    """
    Condition that evaluates to True if the inner condition is False.

    Example:
        negated = ~cond1  # the condition is NOT satisfied
    """
    def __init__(self, cond: PeakCondition):
        self.cond = cond

    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        return ~self.cond.evaluate(peaks)

class IntensityThreshold(PeakCondition):
    """
    Condition that selects peaks with intensity >= a given threshold.

    Example:
        cond = IntensityThreshold(threshold=100.0)
        mask = cond.evaluate(peaks)  # torch.BoolTensor
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        return peaks.intensity >= self.threshold

class TopKIntensity(PeakCondition):
    """
    Condition that selects the top-k peaks by intensity within each spectrum.
    Vectorized implementation without explicit for-loops.
    """
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def evaluate(self, peaks: PeakSeries) -> torch.BoolTensor:
        n_peaks = peaks._data.shape[0]
        mask = torch.zeros(n_peaks, dtype=torch.bool, device=peaks.device)
        if n_peaks == 0:
            return torch.zeros(0, dtype=torch.bool, device=peaks.device)

        # Get sorted indices by intensity (descending)
        _, sorted_idx = peaks.sort_by_intensity(
            ascending=False,
            in_place=False,
            return_index=True
        )

        # Segment boundaries
        offsets = peaks._offsets  # shape [n_segments+1]
        seg_lens = offsets[1:] - offsets[:-1]  # [n_segments]

        # For each segment, how many peaks to keep (min(k, seg_len))
        keep_counts = torch.minimum(
            torch.full_like(seg_lens, self.k), seg_lens
        )

        # --- Build global indices to keep ---
        # repeat segment ids according to keep_counts
        seg_ids = torch.repeat_interleave(
            torch.arange(len(seg_lens), device=peaks.device), keep_counts
        )
        # repeat positions within each segment
        pos_within = torch.cat(
            [torch.arange(c.item(), device=peaks.device) for c in keep_counts]
        )
        # global indices in sorted order
        mask[sorted_idx[offsets[seg_ids] + pos_within]] = True

        return mask
