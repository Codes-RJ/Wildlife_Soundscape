"""
Wildlife acoustic classification package.

This package provides the modular acoustic-classification subsystem for
the Wildlife Soundscape Mapping & Behavior Analysis System.


Architecture
------------
ClassificationInput
    Backend-independent classification input.

    It may contain:

        - handcrafted DSP features
        - processed model waveform
        - both

    Individual ClassifierBackend implementations determine which inputs
    are mandatory.


ClassifierBackend
    Common abstract interface implemented by every classification
    backend.


HeuristicClassifier
    Transparent rule-based broad acoustic classifier using handcrafted
    acoustic features.


HeuristicClassifierBackend
    Adapter exposing HeuristicClassifier through the common
    ClassifierBackend interface.


create_classifier_backend
    Factory responsible for constructing the classifier selected through
    application configuration.


Supported broad acoustic classes
--------------------------------
- Bird
- Insect
- Amphibian
- Mammal
- Noise
- Unknown


Current operational backend
---------------------------
HeuristicClassifierBackend

    requires_features = True
    requires_audio = False


Future backend architecture
---------------------------
Additional implementations may later include:

- pretrained waveform-based bioacoustic models
- BirdNET adapter
- custom machine-learning models
- ensemble classifiers

These can be introduced behind ClassifierBackend without restructuring
the event-processing pipeline.


Scientific scope
----------------
The current heuristic classification layer provides broad
acoustic-pattern estimation.

It must not be interpreted as validated species-level biological
identification.
"""

from __future__ import annotations

from typing import Any


# ======================================================================
# CORE CLASSIFICATION INTERFACE
# ======================================================================


from .base import (
    ClassificationInput,
    ClassifierBackend,
)


# ======================================================================
# CLASSIFICATION MODELS / BASELINE CLASSIFIER
# ======================================================================


from .classifier import (
    AcousticClass,
    ClassificationResult,
    HeuristicClassifier,
)


# ======================================================================
# OPERATIONAL BACKENDS
# ======================================================================


from .heuristic_backend import (
    HeuristicClassifierBackend,
)


# ======================================================================
# LAZY PACKAGE EXPORTS
# ======================================================================


def __getattr__(
    name: str,
) -> Any:
    """
    Lazily expose package-level objects that may depend on several
    classifier modules.

    Why lazy-load the factory?
    --------------------------
    classification.factory may itself import concrete backend classes.

    Importing it eagerly while classification/__init__.py is still being
    initialized can make future classifier additions more vulnerable to
    circular-import problems.

    The lazy export preserves convenient usage:

        from classification import create_classifier_backend

    without forcing factory.py to execute during initial package setup.
    """

    if name == "create_classifier_backend":

        from .factory import (
            create_classifier_backend,
        )

        return create_classifier_backend

    if name == "BirdNETTaxonomy":

        from .birdnet_context import (
            BirdNETTaxonomy,
        )

        return BirdNETTaxonomy

    if name == "BirdNETGeoContext":

        from .birdnet_context import (
            BirdNETGeoContext,
        )

        return BirdNETGeoContext

    raise AttributeError(
        (
            f"module {__name__!r} "
            f"has no attribute {name!r}"
        )
    )


# ======================================================================
# PUBLIC PACKAGE API
# ======================================================================


__all__ = (
    "AcousticClass",
    "BirdNETGeoContext",
    "BirdNETTaxonomy",
    "ClassificationInput",
    "ClassificationResult",
    "ClassifierBackend",
    "HeuristicClassifier",
    "HeuristicClassifierBackend",
    "create_classifier_backend",
)