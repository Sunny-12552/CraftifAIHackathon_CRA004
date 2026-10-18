# SPDX-License-Identifier: Apache-2.0
"""Alert policy: cooldown gating and incident production."""

from __future__ import annotations

from .cooldown import CooldownDecision, CooldownGate, cooldown_key
from .manager import AlertManager, AlertOutcome, IncidentSink, merge_outcomes

__all__ = [
    "AlertManager",
    "AlertOutcome",
    "CooldownDecision",
    "CooldownGate",
    "IncidentSink",
    "cooldown_key",
    "merge_outcomes",
]
