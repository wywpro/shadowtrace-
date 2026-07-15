"""Production fail-closed settings validation (ISSUE-093 §5).

A ``production`` deployment silently running mock sources/tools/disposition
or simulation mode is a security incident, not a warning: ``Settings``
construction must raise ``ConfigurationError`` and prevent the process from
starting.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError


def _base_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "APP_ENV": "production",
        "SOURCE_MODE": "live_edr",
        "TOOL_MODE": "live",
        "DISPOSITION_MODE": "live_xdr",
        "DISPOSITION_ADAPTER_KIND": "http",
        "SIMULATION_ENABLED": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_production_with_all_live_modes_is_accepted() -> None:
    settings = Settings(**_base_kwargs())
    assert settings.app_env == "production"
    assert settings.production_fail_closed_violations() == []


def test_development_allows_mock_and_simulation() -> None:
    settings = Settings(
        APP_ENV="development",
        SOURCE_MODE="mock_xdr",
        TOOL_MODE="mock",
        DISPOSITION_MODE="mock_xdr",
        DISPOSITION_ADAPTER_KIND="mock",
        SIMULATION_ENABLED=True,
    )
    assert settings.production_fail_closed_violations() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"SIMULATION_ENABLED": True},
        {"SOURCE_MODE": "mock_xdr"},
        {"TOOL_MODE": "mock"},
        {"DISPOSITION_MODE": "mock_xdr"},
        {"DISPOSITION_ADAPTER_KIND": "mock"},
    ],
    ids=[
        "simulation_enabled",
        "source_mode_mock",
        "tool_mode_mock",
        "disposition_mode_mock",
        "disposition_adapter_kind_mock",
    ],
)
def test_production_rejects_any_single_mock_or_simulation_mode(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(**_base_kwargs(**overrides))
    assert exc_info.value.error_code == "configuration_error"
    assert exc_info.value.retryable is False


def test_production_rejects_multiple_mock_modes_with_all_violations_listed() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(
            **_base_kwargs(
                SIMULATION_ENABLED=True,
                SOURCE_MODE="mock_xdr",
                TOOL_MODE="mock",
            )
        )
    violations = exc_info.value.details["violations"]
    assert any("simulation_enabled" in v for v in violations)
    assert any("source_mode" in v for v in violations)
    assert any("tool_mode" in v for v in violations)


def test_app_env_matching_is_case_insensitive() -> None:
    with pytest.raises(ConfigurationError):
        Settings(**_base_kwargs(APP_ENV="Production", SIMULATION_ENABLED=True))


def test_staging_env_is_not_subject_to_production_gate() -> None:
    settings = Settings(
        APP_ENV="staging",
        SOURCE_MODE="mock_xdr",
        TOOL_MODE="mock",
        DISPOSITION_MODE="mock_xdr",
        DISPOSITION_ADAPTER_KIND="mock",
        SIMULATION_ENABLED=True,
    )
    assert settings.production_fail_closed_violations() == []
