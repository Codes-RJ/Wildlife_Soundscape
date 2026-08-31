"""Stable console entry points during the package-layout migration."""

from __future__ import annotations


def receiver() -> None:
    """Start the interactive receiver application."""
    from main import main

    main()


def simulator() -> None:
    """Start the three-node protocol simulator."""
    from simulator import main

    main()


def export_events() -> int:
    """Export persisted event records."""
    from tools.export_events import main

    return main()


def export_research() -> int:
    """Export derived research analytics."""
    from tools.export_research_metrics import main

    return main()


def benchmark_gcc() -> None:
    """Run the GCC research-variant benchmark."""
    from tools.benchmark_gcc_variants import main

    main()
