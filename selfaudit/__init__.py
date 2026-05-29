"""Self-Auditing AI — a self-correcting numerical solver with an audit trail."""

from __future__ import annotations

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .auditor import SelfAuditingSolver, Solution, SolveFailed
from .diagnostician import AnomalyDetected, FitDiagnosis, SelfAuditingFitter
from .fitting import FitResult, Model
from .signals import TimeSeries
from .solver import (
    Method,
    NumericalFailure,
    Problem,
    StrategyOutcome,
    brentq,
    default_methods,
    newton,
    secant,
)

__all__ = [
    "Attempt",
    "AuditLog",
    "ExpectationCheck",
    "ReTest",
    "SelfAuditingSolver",
    "SolveFailed",
    "Solution",
    "AnomalyDetected",
    "FitDiagnosis",
    "SelfAuditingFitter",
    "FitResult",
    "Model",
    "TimeSeries",
    "Method",
    "NumericalFailure",
    "Problem",
    "StrategyOutcome",
    "brentq",
    "default_methods",
    "newton",
    "secant",
]
