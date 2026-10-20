"""Simple comparison policies; paper best-response remains a separate milestone."""

from .simple import (
    AlwaysMEC, AlwaysV2V, MinimumEstimatedLatency, OffloadingPolicy,
    UniformRandom, estimated_service_time_s,
)

__all__ = ["AlwaysMEC", "AlwaysV2V", "MinimumEstimatedLatency", "OffloadingPolicy",
           "UniformRandom", "estimated_service_time_s"]
