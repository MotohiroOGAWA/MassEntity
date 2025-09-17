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
