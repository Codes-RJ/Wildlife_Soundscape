from __future__ import annotations

import logging

from config import ClassificationConfig

from .base import ClassifierBackend
from .heuristic_backend import HeuristicClassifierBackend


logger = logging.getLogger(__name__)


# ======================================================================
# BACKEND NAMES
# ======================================================================


HEURISTIC_BACKEND = "heuristic"

PRETRAINED_BACKEND = "pretrained"

BIRDNET_BACKEND = "birdnet"

ENSEMBLE_BACKEND = "ensemble"


IMPLEMENTED_BACKENDS = frozenset(
    {
        HEURISTIC_BACKEND,
    }
)


RESERVED_BACKENDS = frozenset(
    {
        PRETRAINED_BACKEND,
        BIRDNET_BACKEND,
        ENSEMBLE_BACKEND,
    }
)


KNOWN_BACKENDS = (
    IMPLEMENTED_BACKENDS
    | RESERVED_BACKENDS
)


# ======================================================================
# HEURISTIC BACKEND CREATION
# ======================================================================


def _create_heuristic_backend() -> HeuristicClassifierBackend:
    """
    Construct the current operational baseline classifier.

    Keeping construction in one helper prevents fallback and explicit
    heuristic selection from drifting into slightly different behavior.
    """

    return HeuristicClassifierBackend()


# ======================================================================
# CLASSIFIER BACKEND FACTORY
# ======================================================================


def create_classifier_backend(
    config: ClassificationConfig,
) -> ClassifierBackend | None:
    """
    Construct the acoustic-classification backend selected by
    ClassificationConfig.

    Returns
    -------
    ClassifierBackend | None

        None
            Classification is disabled.

        ClassifierBackend
            Operational configured backend, or the heuristic fallback
            when a reserved backend is requested and fallback is enabled.


    Current implementation
    ----------------------
    heuristic

        Transparent DSP-feature-based broad acoustic classifier.


    Reserved future backends
    ------------------------
    pretrained

        Generic trained/pretrained acoustic model.

    birdnet

        BirdNET-specific integration.

    ensemble

        Combination of multiple classifier backends.


    Fallback behavior
    -----------------
    Reserved backends may already appear in configuration before their
    implementations are added.

    When:

        fallback_to_heuristic = True

    the factory returns HeuristicClassifierBackend.

    When:

        fallback_to_heuristic = False

    the factory raises NotImplementedError rather than pretending the
    requested model exists.
    """

    # ==================================================================
    # CONFIGURATION TYPE
    # ==================================================================

    if not isinstance(
        config,
        ClassificationConfig,
    ):

        raise TypeError(
            (
                "config must be a "
                "ClassificationConfig instance"
            )
        )

    # ==================================================================
    # CONFIGURATION VALIDATION
    # ==================================================================
    #
    # AppConfig normally performs this already.
    #
    # Calling validate() again here makes the factory safe when used
    # independently by tests, scripts or future tools.
    # ==================================================================

    config.validate()

    # ==================================================================
    # CLASSIFICATION DISABLED
    # ==================================================================

    if not config.enabled:

        logger.info(
            "Acoustic classification is disabled."
        )

        return None

    # ==================================================================
    # NORMALIZE BACKEND NAME
    # ==================================================================

    backend_name = (
        config.backend
        .strip()
        .lower()
    )

    # ==================================================================
    # HEURISTIC BACKEND
    # ==================================================================

    if (
        backend_name
        == HEURISTIC_BACKEND
    ):

        backend = (
            _create_heuristic_backend()
        )

        logger.info(
            (
                "Classification backend selected "
                "| requested=%s "
                "| active=%s "
                "| version=%s"
            ),
            backend_name,
            backend.name,
            backend.version,
        )

        return backend

    # ==================================================================
    # RESERVED FUTURE BACKENDS
    # ==================================================================

    if (
        backend_name
        in RESERVED_BACKENDS
    ):

        return (
            _handle_unavailable_backend(
                requested_backend=
                    backend_name,

                config=
                    config,
            )
        )

    # ==================================================================
    # UNKNOWN BACKEND
    # ==================================================================
    #
    # ClassificationConfig.validate() should normally make this branch
    # unreachable.
    #
    # It remains intentionally defensive because configuration contracts
    # can change independently in the future.
    # ==================================================================

    known = ", ".join(
        sorted(
            KNOWN_BACKENDS
        )
    )

    raise ValueError(
        (
            "Unsupported classification backend "
            f"'{config.backend}'. "
            f"Known backends: {known}"
        )
    )


# ======================================================================
# FUTURE-BACKEND FALLBACK
# ======================================================================


def _handle_unavailable_backend(
    *,
    requested_backend: str,
    config: ClassificationConfig,
) -> ClassifierBackend:
    """
    Handle a recognized backend whose implementation does not yet exist.

    This intentionally distinguishes:

        configured backend
            what the user requested

    from:

        active backend
            what is actually running

    This prevents research logs or the dashboard from implying that
    BirdNET/pretrained inference occurred when the heuristic fallback was
    actually used.
    """

    requested_backend = (
        str(
            requested_backend
        )
        .strip()
        .lower()
    )

    if (
        requested_backend
        not in RESERVED_BACKENDS
    ):

        raise ValueError(
            (
                "_handle_unavailable_backend() "
                "received a backend that is not "
                "reserved/unavailable: "
                f"'{requested_backend}'"
            )
        )

    # ==================================================================
    # FALLBACK DISABLED
    # ==================================================================

    if not (
        config.fallback_to_heuristic
    ):

        raise NotImplementedError(
            (
                "Classification backend "
                f"'{requested_backend}' is configured "
                "but has not been implemented yet, "
                "and heuristic fallback is disabled."
            )
        )

    # ==================================================================
    # FALLBACK WARNING
    # ==================================================================

    logger.warning(
        (
            "Classification backend '%s' "
            "is recognized but not implemented. "
            "Falling back to the heuristic backend."
        ),
        requested_backend,
    )

    # ==================================================================
    # CREATE FALLBACK
    # ==================================================================

    backend = (
        _create_heuristic_backend()
    )

    logger.info(
        (
            "Classification fallback selected "
            "| requested=%s "
            "| active=%s "
            "| version=%s"
        ),
        requested_backend,
        backend.name,
        backend.version,
    )

    return backend