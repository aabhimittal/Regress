"""Real language-model agents for validating REGRESS outside the simulator."""

from .agent import REGIMES, NoteAgent
from .backends import (
    AnthropicBackend,
    BackendRefusal,
    CachedBackend,
    Completion,
    LlamaCppBackend,
    ReplayBackend,
    RuleBackend,
    ScriptedBackend,
)
from .experiment import RegimeResult, run_experiment, run_regime
from .world import Fact, FactWorld, answer_score

__all__ = [
    "NoteAgent",
    "REGIMES",
    "AnthropicBackend",
    "BackendRefusal",
    "CachedBackend",
    "Completion",
    "LlamaCppBackend",
    "ReplayBackend",
    "RuleBackend",
    "ScriptedBackend",
    "RegimeResult",
    "run_experiment",
    "run_regime",
    "Fact",
    "FactWorld",
    "answer_score",
]
