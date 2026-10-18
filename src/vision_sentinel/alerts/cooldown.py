# SPDX-License-Identifier: Apache-2.0
"""Per-key cooldown gating.

The zone state machine already prevents *frame-level* duplicates. The cooldown
handles the slower case: a person who steps out of a zone and straight back in
should not open a second incident while it is still the same event in progress.

Semantics
---------
* One window per ``(zone, track)`` key.
* The window is opened when an entry is *accepted*, and is deliberately **not**
  cleared when the person leaves. Clearing it on exit would make the cooldown
  meaningless, because every re-entry would immediately qualify.
* Therefore: "at most one entry incident per zone and track per window", which
  is exactly the behaviour the acceptance criteria describe.

Monotonic time is used so a system clock change cannot disable the gate, and the
clock is injectable so the behaviour is testable without sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class CooldownDecision:
    """The outcome of asking the gate about a key."""

    allowed: bool
    remaining_s: float = 0.0


class CooldownGate:
    """Suppresses repeated events for the same key inside a fixed window."""

    def __init__(
        self,
        window_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds < 0:
            raise ValueError("window_seconds must be zero or greater")
        self.window_seconds = window_seconds
        self._clock = clock
        self._last_accepted: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        self._pruned_at = 0.0

    # -- evaluation ------------------------------------------------------
    def peek(self, key: str) -> CooldownDecision:
        """Ask whether ``key`` would be allowed, without recording anything."""
        if self.window_seconds == 0:
            return CooldownDecision(allowed=True)
        last = self._last_accepted.get(key)
        if last is None:
            return CooldownDecision(allowed=True)
        elapsed = self._clock() - last
        remaining = self.window_seconds - elapsed
        if remaining > 0:
            return CooldownDecision(allowed=False, remaining_s=remaining)
        return CooldownDecision(allowed=True)

    def evaluate(self, key: str) -> CooldownDecision:
        """Ask about ``key`` and, when allowed, open a new window for it."""
        decision = self.peek(key)
        if decision.allowed:
            self._last_accepted[key] = self._clock()
        else:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
        self._maybe_prune()
        return decision

    def suppressed_count(self, key: str | None = None) -> int:
        if key is None:
            return sum(self._suppressed.values())
        return self._suppressed.get(key, 0)

    def reset(self) -> None:
        self._last_accepted.clear()
        self._suppressed.clear()

    # -- maintenance -----------------------------------------------------
    def _maybe_prune(self) -> None:
        """Keep memory bounded during long unattended sessions."""
        now = self._clock()
        if now - self._pruned_at < max(self.window_seconds, 60.0):
            return
        self._pruned_at = now
        horizon = now - self.window_seconds * 2
        for key in [key for key, stamp in self._last_accepted.items() if stamp < horizon]:
            del self._last_accepted[key]


def cooldown_key(zone_id: str, track_id: str | None) -> str:
    """Build the canonical cooldown key for a zone and track."""
    return f"{zone_id}::{track_id if track_id is not None else '__anonymous__'}"
