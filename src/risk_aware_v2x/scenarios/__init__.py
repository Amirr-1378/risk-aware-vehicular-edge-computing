"""Seeded synthetic development scenarios; not empirically calibrated traces."""

from .specification import ScenarioSpec, TaskClassSpec
from .generator import GeneratedScenario, generate_scenario

__all__ = ["ScenarioSpec", "TaskClassSpec", "GeneratedScenario", "generate_scenario"]
