"""Tests for the installable package facade and console entry points."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import wildlife_soundscape
from wildlife_soundscape import cli


def test_package_exposes_version() -> None:
    assert wildlife_soundscape.__version__


def test_receiver_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "main",
        SimpleNamespace(main=lambda: calls.append("receiver")),
    )

    cli.receiver()

    assert calls == ["receiver"]


def test_simulator_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "simulator",
        SimpleNamespace(main=lambda: calls.append("simulator")),
    )

    cli.simulator()

    assert calls == ["simulator"]


def test_export_entry_points_return_status(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "tools.export_events",
        SimpleNamespace(main=lambda: 3),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.export_research_metrics",
        SimpleNamespace(main=lambda: 4),
    )

    assert cli.export_events() == 3
    assert cli.export_research() == 4


def test_benchmark_entry_point_delegates(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "tools.benchmark_gcc_variants",
        SimpleNamespace(main=lambda: calls.append("benchmark")),
    )

    cli.benchmark_gcc()

    assert calls == ["benchmark"]
