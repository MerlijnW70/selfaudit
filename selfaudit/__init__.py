"""Self-Auditing AI — a self-correcting numerical solver with an audit trail."""

from __future__ import annotations

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .auditor import SelfAuditingSolver, Solution, SolveFailed
from .datasets import (
    Check,
    CheckResult,
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    infer_checks,
    iqr_outliers,
    load_csv,
    no_missing_required,
    timestamps_monotonic,
    values_in_range,
)
from .datasetscanner import ScanReport, SelfAuditingDatasetScanner
from .diagnostician import AnomalyDetected, FitDiagnosis, SelfAuditingFitter
from .fitting import FitResult, Model
from .llm import (
    AnthropicCaller,
    LLMUnavailable,
    ModelCaller,
    ScriptedCaller,
    Task,
    ValidationResult,
    enable_os_truststore,
    exact_field_validator,
    json_schema_validator,
    load_dotenv,
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
from .sources import (
    SourceUnavailable,
    crypto_prices,
    fetch_csv,
    open_meteo,
    usgs_earthquakes,
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
    "SelfAuditingDatasetScanner",
    "ScanReport",
    "Dataset",
    "Check",
    "CheckResult",
    "load_csv",
    "values_in_range",
    "timestamps_monotonic",
    "no_missing_required",
    "duplicate_rate_below",
    "distribution_stationary",
    "iqr_outliers",
    "infer_checks",
    "open_meteo",
    "usgs_earthquakes",
    "crypto_prices",
    "fetch_csv",
    "SourceUnavailable",
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
    "exact_field_validator",
    "load_dotenv",
    "enable_os_truststore",
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
