"""Named independent RNG streams derived deterministically from a master seed."""

from __future__ import annotations

import hashlib
from random import Random

from ..entities._validation import require_non_negative_int


class RandomStreams:
    """Own persistent ``random.Random`` streams under seed protocol ``v1``.

    A SHA-256 digest of master seed and stream name avoids Python's salted
    hash and makes creation order irrelevant. Suggested names are ``arrivals``,
    ``mobility``, ``shadowing``, ``fading``, ``background``, and ``agent``.
    Extra names may isolate individual links, users, or agent operations.
    """

    def __init__(self, master_seed: int) -> None:
        self._master_seed = require_non_negative_int("master_seed", master_seed)
        self._streams: dict[str, Random] = {}

    def stream(self, name: str) -> Random:
        """Return the same stateful generator on every call with this name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stream name must be a non-empty string")
        if name not in self._streams:
            payload = f"risk-aware-v2x:rng-v1:{self._master_seed}:{name}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
            self._streams[name] = Random(seed)
        return self._streams[name]
