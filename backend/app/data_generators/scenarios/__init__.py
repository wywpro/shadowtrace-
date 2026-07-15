"""Demo scenario registry (ISSUE-011).

``SCENARIO_REGISTRY`` maps fixed scenario IDs to built ``MockXDRScenario``
instances (default seed=42). Rebuild via ``build_scenario`` for other seeds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.data_generators.base import TELEMETRY_FILENAMES
from app.data_generators.scenarios._common import (
    SCENARIO_VARIANTS,
    normalize_variant,
    split_telemetry_by_channel,
)
from app.data_generators.scenarios.account_anomaly_fp import (
    SCENARIO_ID as ACCOUNT_ANOMALY_FP_ID,
)
from app.data_generators.scenarios.account_anomaly_fp import build_account_anomaly_fp
from app.data_generators.scenarios.insider_data_exfiltration import (
    SCENARIO_ID as INSIDER_ID,
)
from app.data_generators.scenarios.insider_data_exfiltration import (
    build_insider_data_exfiltration,
)
from app.data_generators.scenarios.suspicious_domain_access import (
    SCENARIO_ID as DOMAIN_ACCESS_ID,
)
from app.data_generators.scenarios.suspicious_domain_access import (
    build_suspicious_domain_access,
)
from app.mock_xdr.models import MockXDRScenario, ScenarioVariant

SCENARIO_BUILDERS: dict[str, Callable[..., MockXDRScenario]] = {
    INSIDER_ID: build_insider_data_exfiltration,
    ACCOUNT_ANOMALY_FP_ID: build_account_anomaly_fp,
    DOMAIN_ACCESS_ID: build_suspicious_domain_access,
}


def build_scenario(
    scenario_id: str,
    *,
    seed: int = 42,
    variant: ScenarioVariant | str = ScenarioVariant.NORMAL,
) -> MockXDRScenario:
    try:
        builder = SCENARIO_BUILDERS[scenario_id]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIO_BUILDERS))
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {known}") from exc
    scenario = builder(seed=seed, variant=normalize_variant(variant))
    return scenario


# Spec: SCENARIO_REGISTRY: dict[str, MockXDRScenario]
SCENARIO_REGISTRY: dict[str, MockXDRScenario] = {
    scenario_id: build_scenario(scenario_id, seed=42) for scenario_id in SCENARIO_BUILDERS
}


def telemetry_for_scenario(scenario: MockXDRScenario) -> dict[str, list[dict[str, Any]]]:
    """Split ``telemetry_timeline`` into the seven fixed channel files."""
    buckets = split_telemetry_by_channel(scenario.telemetry_timeline)
    missing = [name for name, rows in buckets.items() if not rows]
    if missing:
        raise ValueError(f"scenario {scenario.scenario_id} missing telemetry channels: {missing}")
    return buckets


def write_scenario_artifacts(
    scenario: MockXDRScenario,
    out_dir: Path,
    *,
    write_scenario_json: bool = True,
) -> list[Path]:
    """Write seven telemetry JSON files (+ optional scenario dump) under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for channel, rows in telemetry_for_scenario(scenario).items():
        path = out_dir / TELEMETRY_FILENAMES[channel]
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    if write_scenario_json:
        dump = out_dir / f"{scenario.scenario_id}.scenario.json"
        dump.write_text(
            scenario.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(dump)
    return written


__all__ = [
    "SCENARIO_BUILDERS",
    "SCENARIO_REGISTRY",
    "SCENARIO_VARIANTS",
    "build_scenario",
    "telemetry_for_scenario",
    "write_scenario_artifacts",
]
