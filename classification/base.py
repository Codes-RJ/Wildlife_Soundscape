from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from dsp.features import AcousticFeatures

from .classifier import ClassificationResult


# ======================================================================
# STANDARD CLASSIFIER INPUT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationInput:
    """
    Standard input supplied to an acoustic-classification backend.

    Classification inputs are intentionally backend-independent.

    A backend may use:

        - handcrafted DSP features
        - processed waveform audio
        - both
        - potentially neither, if a future backend uses only external
          or contextual information

    Backend-specific requirements are enforced by ClassifierBackend,
    not by this generic input container.

    Examples
    --------
    Heuristic classifier:

        features = AcousticFeatures(...)
        model_audio = None

    Waveform model:

        features = None
        model_audio = waveform

    Ensemble classifier:

        features = AcousticFeatures(...)
        model_audio = waveform
    """

    # ------------------------------------------------------------------
    # DSP FEATURES
    # ------------------------------------------------------------------
    #
    # Optional by design.
    #
    # The current heuristic classifier requires them, but future
    # waveform-based models must remain usable even if handcrafted
    # feature extraction fails.
    # ------------------------------------------------------------------

    features: AcousticFeatures | None

    # ------------------------------------------------------------------
    # AUDIO INFORMATION
    # ------------------------------------------------------------------

    sample_rate: int

    model_audio: np.ndarray | None = (
        None
    )

    # ------------------------------------------------------------------
    # EVENT CONTEXT
    # ------------------------------------------------------------------

    source_node_id: int | None = (
        None
    )

    detector_event_id: int | None = (
        None
    )

    session_id: int | None = (
        None
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate generic classification-input structure.

        Important
        ---------
        This method validates whether supplied data is structurally
        valid.

        It does NOT decide which inputs are mandatory.

        Mandatory-input decisions belong to the selected
        ClassifierBackend.
        """

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        sample_rate = int(
            self.sample_rate
        )

        if (
            sample_rate
            <= 0
        ):

            raise ValueError(
                (
                    "sample_rate must be "
                    "greater than zero"
                )
            )

        # ==============================================================
        # FEATURE TYPE
        # ==============================================================

        if (
            self.features
            is not None
            and not isinstance(
                self.features,
                AcousticFeatures,
            )
        ):

            raise TypeError(
                (
                    "features must be an "
                    "AcousticFeatures instance "
                    "or None"
                )
            )

        # ==============================================================
        # SOURCE NODE
        # ==============================================================

        if (
            self.source_node_id
            is not None
        ):

            source_node_id = int(
                self.source_node_id
            )

            if (
                source_node_id
                <= 0
            ):

                raise ValueError(
                    (
                        "source_node_id must be "
                        "greater than zero"
                    )
                )

        # ==============================================================
        # DETECTOR EVENT ID
        # ==============================================================

        if (
            self.detector_event_id
            is not None
        ):

            detector_event_id = int(
                self.detector_event_id
            )

            if (
                detector_event_id
                < 0
            ):

                raise ValueError(
                    (
                        "detector_event_id "
                        "cannot be negative"
                    )
                )

        # ==============================================================
        # SESSION ID
        # ==============================================================

        if (
            self.session_id
            is not None
        ):

            session_id = int(
                self.session_id
            )

            if (
                session_id
                < 0
            ):

                raise ValueError(
                    (
                        "session_id "
                        "cannot be negative"
                    )
                )

        # ==============================================================
        # MODEL AUDIO
        # ==============================================================

        if (
            self.model_audio
            is not None
        ):

            audio = np.asarray(
                self.model_audio
            )

            # ----------------------------------------------------------
            # MONO
            # ----------------------------------------------------------

            if (
                audio.ndim
                != 1
            ):

                raise ValueError(
                    (
                        "model_audio must be "
                        "a mono 1-D waveform"
                    )
                )

            # ----------------------------------------------------------
            # NON-EMPTY
            # ----------------------------------------------------------

            if (
                audio.size
                == 0
            ):

                raise ValueError(
                    (
                        "model_audio "
                        "cannot be empty"
                    )
                )

            # ----------------------------------------------------------
            # NUMERIC
            # ----------------------------------------------------------

            if not np.issubdtype(
                audio.dtype,
                np.number,
            ):

                raise TypeError(
                    (
                        "model_audio must "
                        "contain numeric samples"
                    )
                )

            # ----------------------------------------------------------
            # REAL-VALUED
            # ----------------------------------------------------------

            if np.issubdtype(
                audio.dtype,
                np.complexfloating,
            ):

                raise TypeError(
                    (
                        "model_audio must contain "
                        "real-valued audio samples"
                    )
                )

            # ----------------------------------------------------------
            # FINITE
            # ----------------------------------------------------------

            if not np.all(
                np.isfinite(
                    audio
                )
            ):

                raise ValueError(
                    (
                        "model_audio contains "
                        "NaN or infinite samples"
                    )
                )

    # ==================================================================
    # AVAILABILITY HELPERS
    # ==================================================================

    @property
    def has_features(
        self,
    ) -> bool:
        """
        Whether handcrafted acoustic features are available.
        """

        return (
            self.features
            is not None
        )

    @property
    def has_audio(
        self,
    ) -> bool:
        """
        Whether processed waveform audio is available.
        """

        return (
            self.model_audio
            is not None
        )


# ======================================================================
# CLASSIFIER BACKEND INTERFACE
# ======================================================================


class ClassifierBackend(
    ABC
):
    """
    Abstract interface implemented by all acoustic classifiers.

    Current implementation
    ----------------------
    HeuristicClassifierBackend

        requires_features = True
        requires_audio = False

    Future implementations
    ----------------------
    Pretrained waveform model:

        requires_features = False
        requires_audio = True

    Ensemble model:

        may require both

    This abstraction keeps EventPipeline, persistence and dashboard
    logic independent of the selected classification implementation.
    """

    # ==================================================================
    # IDENTITY
    # ==================================================================

    @property
    @abstractmethod
    def name(
        self,
    ) -> str:
        """
        Human-readable backend identifier.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def version(
        self,
    ) -> str:
        """
        Backend/model version identifier.
        """

        raise NotImplementedError

    # ==================================================================
    # BACKEND CAPABILITIES
    # ==================================================================

    @property
    def requires_audio(
        self,
    ) -> bool:
        """
        Whether waveform audio is mandatory for this backend.

        Feature-only classifiers should leave this False.
        """

        return False

    @property
    def requires_features(
        self,
    ) -> bool:
        """
        Whether handcrafted DSP features are mandatory.

        The current heuristic backend requires them.
        """

        return True

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def validate_input(
        self,
        classification_input: ClassificationInput,
    ) -> None:
        """
        Validate that an input satisfies this backend's requirements.

        Generic structural validation is already handled by
        ClassificationInput.__post_init__().

        This method handles backend-specific availability requirements.
        """

        if not isinstance(
            classification_input,
            ClassificationInput,
        ):

            raise TypeError(
                (
                    "classification_input must be "
                    "a ClassificationInput instance"
                )
            )

        # ==============================================================
        # AUDIO REQUIREMENT
        # ==============================================================

        if (
            self.requires_audio
            and not classification_input.has_audio
        ):

            raise ValueError(
                (
                    f"classifier backend "
                    f"'{self.name}' requires "
                    "model_audio"
                )
            )

        # ==============================================================
        # FEATURE REQUIREMENT
        # ==============================================================

        if (
            self.requires_features
            and not classification_input.has_features
        ):

            raise ValueError(
                (
                    f"classifier backend "
                    f"'{self.name}' requires "
                    "acoustic features"
                )
            )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    @abstractmethod
    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:
        """
        Classify one detected acoustic event.

        Implementations must return ClassificationResult so that:

            EventPipeline
            SQLite persistence
            CLI
            dashboard
            research analytics

        remain independent of the concrete classifier implementation.

        Implementations should normally call:

            self.validate_input(classification_input)

        before accessing backend-required data.
        """

        raise NotImplementedError