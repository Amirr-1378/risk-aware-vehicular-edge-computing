"""Seeded synthetic development scenarios; not empirically calibrated traces."""

from .specification import ScenarioSpec, TaskDeadlineSpec
from .generator import GeneratedScenario, generate_scenario

__all__ = ["ScenarioSpec", "TaskDeadlineSpec", "GeneratedScenario", "generate_scenario"]
