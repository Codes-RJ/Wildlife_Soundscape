"""Deployment checks shown before a live hardware acquisition."""

from __future__ import annotations

import importlib.util

from dataclasses import dataclass

from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.core.deployment import deployment_config_path


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One human-readable deployment readiness result."""

    status: str
    title: str
    detail: str


def build_preflight_checks(config: AppConfig) -> list[PreflightCheck]:
    """Return static checks; live node health remains available in receiver logs."""

    checks = [
        PreflightCheck(
            "pass",
            "Receiver endpoint",
            f"Configured for {config.network.host}:{config.network.port}.",
        ),
        PreflightCheck(
            "pass",
            "Array geometry",
            f"Coordinates are configured for nodes {sorted(config.expected_nodes)}.",
        ),
        PreflightCheck(
            "warning" if not config.localization.tdoa_calibration.enabled else "pass",
            "TDOA calibration",
            "Disabled: run a controlled calibration before making field-accuracy claims."
            if not config.localization.tdoa_calibration.enabled
            else "Enabled with a validated timing-bias model.",
        ),
        PreflightCheck(
            "pass" if deployment_config_path().is_file() else "info",
            "Deployment overrides",
            f"Loaded {deployment_config_path()}."
            if deployment_config_path().is_file()
            else "Using committed defaults; copy config.example.json to config.local.json for this site.",
        ),
    ]
    if config.classification.backend in {"birdnet", "ensemble"}:
        installed = importlib.util.find_spec("birdnet") is not None
        checks.append(
            PreflightCheck(
                "pass" if installed else "warning",
                "BirdNET runtime",
                "BirdNET package is available."
                if installed
                else "BirdNET is unavailable; heuristic fallback may be used.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "info",
                "Classifier",
                "Heuristic classifier is active; it does not verify species identity.",
            )
        )
    return checks


def render_preflight_checks(config: AppConfig) -> None:
    """Render the preflight list lazily to keep non-dashboard imports lightweight."""

    import streamlit as st

    with st.sidebar.expander("Hardware Preflight", expanded=False):
        for check in build_preflight_checks(config):
            message = f"{check.title}: {check.detail}"
            if check.status == "pass":
                st.success(message)
            elif check.status == "warning":
                st.warning(message)
            else:
                st.info(message)
        st.caption("Before live capture, confirm all three nodes connect and show healthy clocks in the receiver window.")
