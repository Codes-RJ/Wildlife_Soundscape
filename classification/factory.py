from __future__ import annotations

import logging

from config import ClassificationConfig

from .base import ClassifierBackend
from .heuristic_backend import HeuristicClassifierBackend


logger = logging.getLogger(__name__)


# ======================================================================
# CLASSIFIER BACKEND FACTORY
# ======================================================================


def create_classifier_backend(
    config: ClassificationConfig,
) -> ClassifierBackend | None:
    """
    Create the acoustic-classification backend selected in configuration.

    Parameters
    ----------
    config:
        Classification configuration from AppConfig.

    Returns
    -------
    ClassifierBackend | None
        Configured classifier backend.

        None is returned when classification is disabled.

    Current implementation
    ----------------------
    heuristic
        Uses the transparent DSP-feature-based heuristic baseline.

    Reserved future backends
    ------------------------
    pretrained
        Generic trained/pretrained bioacoustic model.

    birdnet
        BirdNET-specific adapter.

    ensemble
        Combination of multiple classifier backends.

    Fallback behavior
    -----------------
    Future backends may exist in configuration before their concrete
    implementation is added.

    If fallback_to_heuristic is True, such a backend automatically
    falls back to HeuristicClassifierBackend.

    Otherwise an explicit NotImplementedError is raised.
    """

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

    if backend_name == "heuristic":

        backend = (
            HeuristicClassifierBackend()
        )

        logger.info(
            (
                "Classification backend selected: "
                "%s v%s"
            ),
            backend.name,
            backend.version,
        )

        return backend

    # ==================================================================
    # PRETRAINED MODEL
    # ==================================================================

    if backend_name == "pretrained":

        return _handle_unavailable_backend(
            requested_backend=
                backend_name,

            config=
                config,
        )

    # ==================================================================
    # BIRDNET
    # ==================================================================

    if backend_name == "birdnet":

        return _handle_unavailable_backend(
            requested_backend=
                backend_name,

            config=
                config,
        )

    # ==================================================================
    # ENSEMBLE
    # ==================================================================

    if backend_name == "ensemble":

        return _handle_unavailable_backend(
            requested_backend=
                backend_name,

            config=
                config,
        )

    # ==================================================================
    # UNKNOWN BACKEND
    # ==================================================================
    #
    # ClassificationConfig.validate() should normally catch this before
    # the factory is reached. This guard remains here so the factory is
    # safe when called independently.
    # ==================================================================

    raise ValueError(
        (
            "Unsupported classification backend: "
            f"'{config.backend}'"
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
    Handle a configured backend whose implementation is not yet present.

    This lets configuration reserve future backend names without
    pretending that those models are already implemented.
    """

    if not config.fallback_to_heuristic:

        raise NotImplementedError(
            (
                "Classification backend "
                f"'{requested_backend}' is configured but has not "
                "been implemented yet, and heuristic fallback is "
                "disabled."
            )
        )

    logger.warning(
        (
            "Classification backend '%s' is not implemented yet. "
            "Falling back to the heuristic classifier."
        ),
        requested_backend,
    )

    backend = (
        HeuristicClassifierBackend()
    )

    logger.info(
        (
            "Classification fallback backend: "
            "%s v%s"
        ),
        backend.name,
        backend.version,
    )

    return backend