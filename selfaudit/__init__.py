"""Self-Auditing AI — a self-correcting numerical solver with an audit trail."""

from __future__ import annotations

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .auditor import SelfAuditingSolver, Solution, SolveFailed
from .diagnostician import AnomalyDetected, FitDiagnosis, SelfAuditingFitter
from .fitting import FitResult, Model
from .llm import (
    AnthropicCaller,
    LLMUnavailable,
    ModelCaller,
    ScriptedCaller,
    Task,
    ValidationResult,
    json_schema_validator,
)
from .llmauditor import SelfAuditingValidator, Validation, ValidationFailed
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
    "SelfAuditingValidator",
    "Validation",
    "ValidationFailed",
    "AnthropicCaller",
    "ScriptedCaller",
    "ModelCaller",
    "Task",
    "ValidationResult",
    "LLMUnavailable",
    "json_schema_validator",
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
