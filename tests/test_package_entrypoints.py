"""Tests for the installable package facade and console entry points."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import wildlife_soundscape.runtime.main as receiver_main
import wildlife_soundscape
from wildlife_soundscape import cli


def test_package_exposes_version() -> None:
    assert wildlife_soundscape.__version__


def test_receiver_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.runtime.main",
        SimpleNamespace(main=lambda: calls.append("receiver")),
    )

    cli.receiver()

    assert calls == ["receiver"]


def test_simulator_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.runtime.simulator",
        SimpleNamespace(main=lambda: calls.append("simulator")),
    )

    cli.simulator()

    assert calls == ["simulator"]


def test_export_entry_points_return_status(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.tools.export_events",
        SimpleNamespace(main=lambda: 3),
    )
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.tools.export_research_metrics",
        SimpleNamespace(main=lambda: 4),
    )

    assert cli.export_events() == 3
    assert cli.export_research() == 4


def test_benchmark_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.tools.benchmark_gcc_variants",
        SimpleNamespace(main=lambda: calls.append("benchmark")),
    )

    cli.benchmark_gcc()

    assert calls == ["benchmark"]


def test_dataset_validator_entry_point_returns_status(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wildlife_soundscape.tools.validate_dataset",
        SimpleNamespace(main=lambda: 5),
    )

    assert cli.validate_dataset() == 5


def test_receiver_parser_supports_automatic_acquisition() -> None:
    args = receiver_main.build_argument_parser().parse_args(
        ["--auto-start", "--session-label", "demo"]
    )

    assert args.auto_start is True
    assert args.session_label == "demo"


def test_automatic_acquisition_starts_after_nodes_are_ready() -> None:
    connection = SimpleNamespace(
        state=SimpleNamespace(connected=True),
        writer=SimpleNamespace(is_closing=lambda: False),
    )
    server = SimpleNamespace(
        config=SimpleNamespace(expected_nodes=(1, 2, 3)),
        connections={1: connection, 2: connection, 3: connection},
        start_acquisition=AsyncMock(return_value=0x1234),
    )

    session_id = asyncio.run(receiver_main.start_acquisition_when_ready(server, "demo"))

    assert session_id == 0x1234
    server.start_acquisition.assert_awaited_once_with("demo")
