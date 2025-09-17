from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .MSDataset import MSDataset


class SpecCondition(ABC):
    """
    Abstract base class for a condition to evaluate on spectra (MSDataset).
    Evaluates all spectra at once, returning a boolean mask of shape [n_spectra].
    """

    @abstractmethod
    def evaluate(self, ds: "MSDataset") -> torch.BoolTensor:
        """
        Evaluate condition across all spectra in the dataset.

        Args:
            ds (MSDataset): Dataset containing spectra.

        Returns:
            torch.BoolTensor: Boolean mask of shape [n_spectra], True for spectra kept.
        """
        pass

    def __and__(self, other: "SpecCondition") -> "SpecCondition":
        return AndSpecCondition(self, other)

    def __or__(self, other: "SpecCondition") -> "SpecCondition":
        return OrSpecCondition(self, other)

    def __invert__(self) -> "SpecCondition":
        return NotSpecCondition(self)


class AndSpecCondition(SpecCondition):
    def __init__(self, cond1: SpecCondition, cond2: SpecCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, ds: "MSDataset") -> torch.BoolTensor:
        return self.cond1.evaluate(ds) & self.cond2.evaluate(ds)


class OrSpecCondition(SpecCondition):
    def __init__(self, cond1: SpecCondition, cond2: SpecCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, ds: "MSDataset") -> torch.BoolTensor:
        return self.cond1.evaluate(ds) | self.cond2.evaluate(ds)


class NotSpecCondition(SpecCondition):
    def __init__(self, cond: SpecCondition):
        self.cond = cond

    def evaluate(self, ds: "MSDataset") -> torch.BoolTensor:
        return ~self.cond.evaluate(ds)

class AllIntegerMZ(SpecCondition):
    """
    Select spectra where all m/z values are integers.
    """

    def evaluate(self, ds: MSDataset) -> torch.BoolTensor:
        offsets = ds.peak_series._offsets
        mz = ds.peak_series.mz

        # Check if each m/z is an integer
        is_int = (mz % 1 == 0)

        # Aggregate across each segment: True only if all peaks are integers
        seg_ids = torch.repeat_interleave(
            torch.arange(len(offsets) - 1, device=mz.device),
            offsets[1:] - offsets[:-1]
        )
        result = torch.ones(len(offsets) - 1, dtype=torch.bool, device=mz.device)
        result = result.scatter_reduce(0, seg_ids, is_int, reduce="amin", include_self=True)

        return result
