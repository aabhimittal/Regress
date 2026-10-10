"""REGRESS — regression testing for self-evolving agents."""

from .bisect import BisectError, BisectResult, Onset, bisect_checkpoints, estimate_onset, locate_onset
from .core import Agent, FunctionAgent, MemorySnapshot, Probe, Response, exact_match
from .detector import ALARM, OK, WATCH, CollapseDetector, Signal, Thresholds
from .monitor import CycleReport, ProbeSuiteMismatch, RegressionMonitor
from .probes import ProbeSuite, ProbeSuiteTampered

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "FunctionAgent",
    "MemorySnapshot",
    "Probe",
    "Response",
    "exact_match",
    "ProbeSuite",
    "ProbeSuiteTampered",
    "ProbeSuiteMismatch",
    "RegressionMonitor",
    "CycleReport",
    "CollapseDetector",
    "Thresholds",
    "Signal",
    "OK",
    "WATCH",
    "ALARM",
    "bisect_checkpoints",
    "BisectResult",
    "BisectError",
    "estimate_onset",
    "locate_onset",
    "Onset",
]
