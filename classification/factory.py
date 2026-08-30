from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import importlib.util
import logging


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================


from config import (
    ClassificationConfig,
)


# ======================================================================
# CLASSIFICATION CONTRACT
# ======================================================================


from .base import (
    ClassifierBackend,
)


from .heuristic_backend import (
    HeuristicClassifierBackend,
)


# ======================================================================
# LOGGER
# ======================================================================


logger = logging.getLogger(
    __name__
)


# ======================================================================
# BACKEND NAMES
# ======================================================================


HEURISTIC_BACKEND = (
    "heuristic"
)


PRETRAINED_BACKEND = (
    "pretrained"
)


BIRDNET_BACKEND = (
    "birdnet"
)


ENSEMBLE_BACKEND = (
    "ensemble"
)


# ======================================================================
# BACKEND IMPLEMENTATION STATE
# ======================================================================
#
# "Implemented" means source-code support exists in this repository.
#
# BirdNET and Ensemble remain optional at runtime because BirdNET's
# external model/runtime dependencies are intentionally not required by
# the core Wildlife Soundscape installation.
# ======================================================================


IMPLEMENTED_BACKENDS = frozenset(
    {
        HEURISTIC_BACKEND,
        BIRDNET_BACKEND,
        ENSEMBLE_BACKEND,
    }
)


RESERVED_BACKENDS = frozenset(
    {
        PRETRAINED_BACKEND,
    }
)


KNOWN_BACKENDS = (
    IMPLEMENTED_BACKENDS
    | RESERVED_BACKENDS
)


# ======================================================================
# OPTIONAL DEPENDENCIES
# ======================================================================


BIRDNET_PACKAGE_NAME = (
    "birdnet"
)


BIRDNET_ONNX_MODULE_NAME = (
    "onnxruntime"
)


# ======================================================================
# ENSEMBLE DEFAULTS
# ======================================================================
#
# Equal weights are deliberately used initially.
#
# We should not assign higher authority to BirdNET or the heuristic
# classifier until controlled validation data provides evidence for a
# different weighting.
# ======================================================================


ENSEMBLE_HEURISTIC_WEIGHT = (
    1.0
)


ENSEMBLE_BIRDNET_WEIGHT = (
    1.0
)


# ======================================================================
# HEURISTIC BACKEND CREATION
# ======================================================================


def _create_heuristic_backend() -> HeuristicClassifierBackend:
    """
    Construct the transparent baseline classifier.

    Keeping construction in one helper ensures that:

        explicit heuristic selection

    and:

        heuristic fallback

    create exactly the same backend implementation.
    """

    return (
        HeuristicClassifierBackend()
    )


# ======================================================================
# OPTIONAL MODULE AVAILABILITY
# ======================================================================


def _module_available(
    module_name: str,
) -> bool:
    """
    Return whether an importable Python module is available.

    This performs module discovery only.

    It does not import or initialize the package.
    """

    if not isinstance(
        module_name,
        str,
    ):

        raise TypeError(
            (
                "module_name must "
                "be a string."
            )
        )

    module_name = (
        module_name.strip()
    )

    if not (
        module_name
    ):

        raise ValueError(
            (
                "module_name cannot "
                "be empty."
            )
        )

    try:

        specification = (
            importlib.util.find_spec(
                module_name
            )
        )

    except (
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ):

        return (
            False
        )

    return (
        specification
        is not None
    )


# ======================================================================
# BIRDNET DEPENDENCY CHECK
# ======================================================================


def _birdnet_dependency_problem() -> str | None:
    """
    Return a human-readable BirdNET dependency problem.

    None means the dependencies required by the currently implemented
    BirdNET V3 ONNX backend appear importable.

    Important
    ---------
    This check intentionally does not load the BirdNET neural network.

    Model initialization remains lazy inside BirdNETClassifierBackend.
    """

    if not (
        _module_available(
            BIRDNET_PACKAGE_NAME
        )
    ):

        return (
            "the optional 'birdnet' package "
            "is not installed"
        )

    # ------------------------------------------------------------------
    # Current project BirdNET backend uses:
    #
    #     acoustic V3.0
    #     ONNX
    #
    # The selected implementation therefore also requires an importable
    # ONNX Runtime.
    # ------------------------------------------------------------------

    if not (
        _module_available(
            BIRDNET_ONNX_MODULE_NAME
        )
    ):

        return (
            "BirdNET is installed, but the "
            "ONNX runtime required by the "
            "configured BirdNET V3 backend "
            "is unavailable"
        )

    return (
        None
    )


# ======================================================================
# BIRDNET BACKEND CREATION
# ======================================================================


def _create_birdnet_backend(
    config: ClassificationConfig,
) -> ClassifierBackend:
    """
    Construct the optional BirdNET backend.

    BirdNET-specific source code is imported lazily so selecting the
    heuristic backend does not load the BirdNET adapter unnecessarily.


    Configuration mapping
    ---------------------
    ClassificationConfig.model_min_confidence
        ->
    BirdNET minimum species confidence.

    ClassificationConfig.top_k
        ->
    Number of strongest species predictions retained.


    Audio
    -----
    BirdNET requires model_audio.

    Therefore:

        provide_model_audio = True

    is mandatory when BirdNET is active.


    Sampling
    --------
    The project's acquisition sample rate remains independent from the
    model's internal sample-rate requirements.
    """

    # ==================================================================
    # AUDIO CONTRACT
    # ==================================================================

    if not (
        config.provide_model_audio
    ):

        raise ValueError(
            (
                "BirdNET requires waveform model audio, "
                "but ClassificationConfig."
                "provide_model_audio is False."
            )
        )

    # ==================================================================
    # EXTERNAL DEPENDENCY
    # ==================================================================

    dependency_problem = (
        _birdnet_dependency_problem()
    )

    if (
        dependency_problem
        is not None
    ):

        raise RuntimeError(
            (
                "BirdNET backend cannot be activated because "
                f"{dependency_problem}."
            )
        )

    # ==================================================================
    # LAZY PROJECT IMPORT
    # ==================================================================

    from .birdnet_backend import (
        BirdNETClassifierBackend,
    )

    # ==================================================================
    # GENERIC MODEL CONFIGURATION WARNINGS
    # ==================================================================
    #
    # model_path, labels_path and model_sample_rate were introduced for
    # generic trained-model backends.
    #
    # BirdNET manages its own model assets and model-input conversion, so
    # those generic configuration values are not interpreted as BirdNET
    # overrides.
    # ==================================================================

    if (
        config.model_path
        is not None
    ):

        logger.warning(
            (
                "Classification model_path is configured, "
                "but the BirdNET backend uses its own "
                "model loader and will not use "
                "ClassificationConfig.model_path."
            )
        )

    if (
        config.labels_path
        is not None
    ):

        logger.warning(
            (
                "Classification labels_path is configured, "
                "but the BirdNET backend uses BirdNET's "
                "taxonomy and will not use "
                "ClassificationConfig.labels_path."
            )
        )

    if (
        config.model_sample_rate
        is not None
    ):

        logger.warning(
            (
                "Classification model_sample_rate is configured, "
                "but the BirdNET backend manages model-specific "
                "sample-rate conversion internally."
            )
        )

    # ==============================================================
    # TAXONOMY & GEOGRAPHIC CONTEXT
    # ==============================================================

    from .birdnet_context import (
        BirdNETGeoContext,
        BirdNETTaxonomy,
    )

    taxonomy = (
        BirdNETTaxonomy.from_csv(config.taxonomy_path)
        if config.taxonomy_path is not None
        else None
    )

    geo_context = (
        BirdNETGeoContext(
            enabled=config.use_geo_filter,
            latitude=config.latitude,
            longitude=config.longitude,
            week=config.week,
            min_confidence=config.geo_min_confidence,
        )
        if config.use_geo_filter
        else None
    )

    # ==============================================================
    # CONSTRUCTION
    # ==============================================================

    return (
        BirdNETClassifierBackend(
            min_confidence=
                config.model_min_confidence,

            top_k=
                config.top_k,

            taxonomy=
                taxonomy,

            geo_context=
                geo_context,
        )
    )


# ======================================================================
# ENSEMBLE BACKEND CREATION
# ======================================================================


def _create_ensemble_backend(
    config: ClassificationConfig,
) -> ClassifierBackend:
    """
    Construct the currently implemented ensemble classifier.

    Current members
    ---------------
    1. HeuristicClassifierBackend

    2. BirdNETClassifierBackend


    Fusion
    ------
    The ensemble performs weighted fusion over the common broad
    AcousticClass ontology.

    Current weights:

        heuristic = 1.0
        BirdNET   = 1.0


    Why equal weights
    -----------------
    No controlled project-specific benchmark currently demonstrates
    that either classifier should receive greater voting authority.

    Weight optimization should therefore be performed only after labeled
    evaluation data exists.


    Runtime degradation
    -------------------
    When:

        fallback_to_heuristic = True

    the ensemble permits a member to fail during one event and can still
    return a result from the remaining successful member.

    When:

        fallback_to_heuristic = False

    every configured ensemble member is required to succeed.


    Construction dependency
    -----------------------
    BirdNET must be available when the ensemble is initially created.

    If BirdNET cannot even be constructed, the factory handles the
    ensemble as an unavailable optional backend and applies the normal
    fallback policy.
    """

    # ==================================================================
    # AUDIO CONTRACT
    # ==================================================================
    #
    # The heuristic member requires features.
    #
    # BirdNET requires waveform audio.
    #
    # The ensemble therefore requires both data paths.
    # ==================================================================

    if not (
        config.provide_model_audio
    ):

        raise ValueError(
            (
                "The current ensemble contains BirdNET "
                "and therefore requires "
                "ClassificationConfig."
                "provide_model_audio=True."
            )
        )

    # ==================================================================
    # MEMBER CONSTRUCTION
    # ==================================================================

    heuristic_backend = (
        _create_heuristic_backend()
    )

    birdnet_backend = (
        _create_birdnet_backend(
            config
        )
    )

    # ==================================================================
    # LAZY ENSEMBLE IMPORT
    # ==================================================================
    #
    # Avoid importing ensemble implementation when another backend is
    # selected.
    # ==================================================================

    from .ensemble_backend import (
        EnsembleClassifierBackend,
        EnsembleMember,
    )

    # ==================================================================
    # RUNTIME FAILURE POLICY
    # ==================================================================
    #
    # fallback_to_heuristic=True
    #
    #     One successful ensemble member is enough. If BirdNET fails
    #     during inference, heuristic evidence may still be returned and
    #     the BirdNET failure remains recorded in result reasons.
    #
    # fallback_to_heuristic=False
    #
    #     Both members must complete successfully.
    # ==================================================================

    if (
        config.fallback_to_heuristic
    ):

        continue_on_member_error = (
            True
        )

        minimum_successful_members = (
            1
        )

    else:

        continue_on_member_error = (
            False
        )

        minimum_successful_members = (
            2
        )

    # ==================================================================
    # CONSTRUCTION
    # ==================================================================

    return (
        EnsembleClassifierBackend(
            members=(
                EnsembleMember(
                    backend=
                        heuristic_backend,

                    weight=
                        ENSEMBLE_HEURISTIC_WEIGHT,

                    label=
                        HEURISTIC_BACKEND,
                ),

                EnsembleMember(
                    backend=
                        birdnet_backend,

                    weight=
                        ENSEMBLE_BIRDNET_WEIGHT,

                    label=
                        BIRDNET_BACKEND,
                ),
            ),

            continue_on_member_error=
                continue_on_member_error,

            min_successful_members=
                minimum_successful_members,
        )
    )


# ======================================================================
# FALLBACK TO HEURISTIC
# ======================================================================


def _fallback_to_heuristic(
    *,
    requested_backend: str,
    reason: str,
) -> HeuristicClassifierBackend:
    """
    Construct heuristic fallback while clearly logging why it is active.

    This distinction is important for research traceability.

    A fallback classification must never be logged or persisted as if
    BirdNET, Ensemble or another requested model actually produced it.
    """

    backend = (
        _create_heuristic_backend()
    )

    logger.warning(
        (
            "Classification backend fallback "
            "| requested=%s "
            "| active=%s "
            "| reason=%s"
        ),
        requested_backend,
        backend.name,
        reason,
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

    return (
        backend
    )


# ======================================================================
# OPTIONAL IMPLEMENTED BACKEND FAILURE
# ======================================================================


def _handle_optional_backend_failure(
    *,
    requested_backend: str,
    config: ClassificationConfig,
    error: Exception,
) -> ClassifierBackend:
    """
    Handle an implemented optional backend that cannot be activated.

    This differs from `_handle_unavailable_backend()`.


    Optional implemented backend
    ----------------------------
    Source code exists, but an optional dependency or required
    configuration is unavailable.

    Examples:

        BirdNET dependency unavailable

        Ensemble cannot construct BirdNET member


    Reserved backend
    ----------------
    Source implementation does not yet exist.
    """

    if not isinstance(
        error,
        Exception,
    ):

        raise TypeError(
            (
                "error must be "
                "an Exception."
            )
        )

    reason = (
        str(
            error
        )
        .strip()
        or type(
            error
        ).__name__
    )

    # ==================================================================
    # FALLBACK ENABLED
    # ==================================================================

    if (
        config.fallback_to_heuristic
    ):

        return (
            _fallback_to_heuristic(
                requested_backend=
                    requested_backend,

                reason=
                    reason,
            )
        )

    # ==================================================================
    # FALLBACK DISABLED
    # ==================================================================

    raise RuntimeError(
        (
            "Classification backend "
            f"'{requested_backend}' is implemented "
            "but could not be activated, and "
            "heuristic fallback is disabled. "
            f"Reason: {reason}"
        )
    ) from error


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
            Active classifier implementation.


    Implemented backends
    --------------------
    heuristic

        Transparent DSP-feature-based broad acoustic classifier.


    birdnet

        Optional species-recognition classifier adapted to the project's
        broad AcousticClass output contract.


    ensemble

        Weighted fusion of:

            heuristic
            BirdNET

        while preserving individual backend evidence.


    Reserved backend
    ----------------
    pretrained

        Generic project-specific trained/pretrained classifier.


    Fallback policy
    ---------------
    fallback_to_heuristic=True

        If an optional backend cannot be activated, the heuristic
        classifier becomes active.

        The ensemble may also continue when one runtime member fails.


    fallback_to_heuristic=False

        Optional backend activation errors are surfaced.

        The ensemble requires all configured members to succeed.


    Research traceability
    ---------------------
    The active backend always reports its own:

        name
        version

    Persistence therefore records what actually performed inference,
    rather than merely the backend originally requested in configuration.
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
                "ClassificationConfig instance."
            )
        )

    # ==================================================================
    # CONFIGURATION VALIDATION
    # ==================================================================
    #
    # AppConfig normally performs this already.
    #
    # Re-validating keeps the factory safe when independently used by:
    #
    #     tests
    #     scripts
    #     notebooks
    #     research tooling
    # ==================================================================

    config.validate()

    # ==================================================================
    # CLASSIFICATION DISABLED
    # ==================================================================

    if not (
        config.enabled
    ):

        logger.info(
            (
                "Acoustic classification "
                "is disabled."
            )
        )

        return (
            None
        )

    # ==================================================================
    # NORMALIZE BACKEND NAME
    # ==================================================================

    backend_name = (
        config.backend
        .strip()
        .lower()
    )

    # ==================================================================
    # HEURISTIC
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

        return (
            backend
        )

    # ==================================================================
    # BIRDNET
    # ==================================================================

    if (
        backend_name
        == BIRDNET_BACKEND
    ):

        try:

            backend = (
                _create_birdnet_backend(
                    config
                )
            )

        except (
            FileNotFoundError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:

            return (
                _handle_optional_backend_failure(
                    requested_backend=
                        BIRDNET_BACKEND,

                    config=
                        config,

                    error=
                        exc,
                )
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

        return (
            backend
        )

    # ==================================================================
    # ENSEMBLE
    # ==================================================================

    if (
        backend_name
        == ENSEMBLE_BACKEND
    ):

        try:

            backend = (
                _create_ensemble_backend(
                    config
                )
            )

        except (
            FileNotFoundError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:

            return (
                _handle_optional_backend_failure(
                    requested_backend=
                        ENSEMBLE_BACKEND,

                    config=
                        config,

                    error=
                        exc,
                )
            )

        logger.info(
            (
                "Classification backend selected "
                "| requested=%s "
                "| active=%s "
                "| version=%s "
                "| members=heuristic,birdnet"
            ),
            backend_name,
            backend.name,
            backend.version,
        )

        return (
            backend
        )

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
    # It remains defensive because configuration contracts may evolve
    # independently from the factory.
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
# RESERVED-BACKEND FALLBACK
# ======================================================================


def _handle_unavailable_backend(
    *,
    requested_backend: str,
    config: ClassificationConfig,
) -> ClassifierBackend:
    """
    Handle a recognized backend whose source implementation does not yet
    exist.

    This intentionally distinguishes:

        configured backend
            what was requested

    from:

        active backend
            what is actually executing inference

    so logs and persisted classifier metadata cannot imply that an
    unavailable model actually ran.
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
    # FALLBACK ENABLED
    # ==================================================================

    return (
        _fallback_to_heuristic(
            requested_backend=
                requested_backend,

            reason=
                (
                    "backend source implementation "
                    "does not exist yet"
                ),
        )
    )