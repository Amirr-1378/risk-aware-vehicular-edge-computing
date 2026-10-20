"""Explicit exploration scheduling by elapsed physical training slots."""

from dataclasses import dataclass

from ..entities._validation import require_fraction, require_non_negative_int, require_positive_int


@dataclass(frozen=True, slots=True)
class EpsilonSchedule:
    initial: float = 1.
    final: float = .05
    decay_slots: int = 10_000

    def __post_init__(self):
        object.__setattr__(self, "initial", require_fraction("initial epsilon", self.initial))
        object.__setattr__(self, "final", require_fraction("final epsilon", self.final))
        require_positive_int("decay_slots", self.decay_slots)
        if self.final > self.initial:
            raise ValueError("epsilon must not increase during decay")

    def value(self, elapsed_training_slots: int) -> float:
        slots = require_non_negative_int("elapsed_training_slots", elapsed_training_slots)
        if slots >= self.decay_slots:
            return self.final
        fraction = slots / self.decay_slots
        return self.initial + fraction * (self.final - self.initial)
