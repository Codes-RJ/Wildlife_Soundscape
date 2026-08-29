"""
Wildlife acoustic classification package.

This package provides the modular acoustic-classification subsystem for
the Wildlife Soundscape project.

Classification architecture
---------------------------
ClassificationInput
    Standardized event input containing DSP features, optional model
    waveform data, sample-rate information, and event metadata.

ClassifierBackend
    Common interface implemented by every classification backend.

HeuristicClassifier
    Transparent rule-based broad acoustic classifier.

HeuristicClassifierBackend
    Adapter exposing HeuristicClassifier through ClassifierBackend.

create_classifier_backend
    Factory responsible for selecting and constructing the configured
    classification backend.

Supported broad acoustic classes
--------------------------------
- Bird
- Insect
- Amphibian
- Mammal
- Noise
- Unknown

Current default
---------------
The current operational backend is the heuristic classifier.

Future architecture
-------------------
Additional backends may later include:

- pretrained bioacoustic models
- BirdNET
- custom machine-learning models
- ensemble classifiers

These can be introduced behind the same ClassifierBackend interface
without restructuring the event-processing pipeline.

Important
---------
Current classification represents broad acoustic-pattern estimation.

It does NOT constitute species-level biological identification.
"""

from .base import (
    ClassificationInput,
    ClassifierBackend,
)

from .classifier import (
    AcousticClass,
    ClassificationResult,
    HeuristicClassifier,
)

from .heuristic_backend import (
    HeuristicClassifierBackend,
)

from .factory import (
    create_classifier_backend,
)


__all__ = (
    "AcousticClass",
    "ClassificationInput",
    "ClassificationResult",
    "ClassifierBackend",
    "HeuristicClassifier",
    "HeuristicClassifierBackend",
    "create_classifier_backend",
)