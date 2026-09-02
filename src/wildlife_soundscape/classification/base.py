from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from dataclasses import (
    dataclass,
)

import numpy as np

from wildlife_soundscape.dsp.features import (
    AcousticFeatures,
)

from .classifier import (
    ClassificationResult,
)


# ======================================================================
# CONSTANTS
# ======================================================================


UINT8_MAX = (
    0xFF
)


UINT32_MAX = (
    0xFFFFFFFF
)


# ======================================================================
# INTEGER VALIDATION
# ======================================================================


def _require_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require one strict Python integer.

    bool is deliberately rejected even though bool is a subclass of int.
    """

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            (
                f"{name} must be "
                "an integer."
            )
        )

    return value


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require one integer greater than zero.
    """

    value = (
        _require_integer(
            value,
            name=
                name,
        )
    )

    if (
        value
        <= 0
    ):

        raise ValueError(
            (
                f"{name} must be "
                "greater than zero."
            )
        )

    return value


def _require_uint8_node_id(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a Protocol-v4 node identifier.

    Node IDs are transported as uint8 and zero is reserved as an invalid
    acoustic-node identifier in this project.
    """

    value = (
        _require_integer(
            value,
            name=
                name,
        )
    )

    if not (
        1
        <= value
        <= UINT8_MAX
    ):

        raise ValueError(
            (
                f"{name} must lie "
                "between 1 and 255."
            )
        )

    return value


def _require_uint32(
    value: int,
    *,
    name: str,
    allow_zero: bool,
) -> int:
    """
    Validate one unsigned 32-bit identifier.
    """

    value = (
        _require_integer(
            value,
            name=
                name,
        )
    )

    minimum = (
        0
        if allow_zero
        else 1
    )

    if not (
        minimum
        <= value
        <= UINT32_MAX
    ):

        if allow_zero:

            raise ValueError(
                (
                    f"{name} must lie between "
                    "0 and 4294967295."
                )
            )

        raise ValueError(
            (
                f"{name} must lie between "
                "1 and 4294967295."
            )
        )

    return value


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

    A backend may consume:

        - handcrafted DSP features
        - processed waveform audio
        - both
        - potentially neither for a future contextual backend

    Backend-specific availability requirements are enforced by
    ClassifierBackend rather than by this generic container.


    Examples
    --------
    Heuristic backend:

        features = AcousticFeatures(...)
        model_audio = None

    Waveform backend:

        features = None
        model_audio = waveform

    Ensemble backend:

        features = AcousticFeatures(...)
        model_audio = waveform


    Design rule
    -----------
    This object validates structural correctness.

    It does not decide whether a particular backend requires features,
    waveform audio, or both.
    """

    # ==================================================================
    # DSP FEATURES
    # ==================================================================
    #
    # Optional by design.
    #
    # The heuristic backend requires features, but future waveform-based
    # classifiers must remain usable even when handcrafted feature
    # extraction is unavailable.
    # ==================================================================

    features: (
        AcousticFeatures
        | None
    )

    # ==================================================================
    # AUDIO INFORMATION
    # ==================================================================

    sample_rate: int

    model_audio: (
        np.ndarray
        | None
    ) = None

    # ==================================================================
    # EVENT CONTEXT
    # ==================================================================

    source_node_id: (
        int
        | None
    ) = None

    detector_event_id: (
        int
        | None
    ) = None

    session_id: (
        int
        | None
    ) = None

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate generic classifier-input structure.

        Backend-specific mandatory inputs are intentionally not enforced
        here. They are handled by ClassifierBackend.validate_input().
        """

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        _require_positive_integer(
            self.sample_rate,
            name=
                "sample_rate",
        )

        # ==============================================================
        # FEATURES
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
                    "or None."
                )
            )

        # ==============================================================
        # SOURCE NODE
        # ==============================================================

        if (
            self.source_node_id
            is not None
        ):

            _require_uint8_node_id(
                self.source_node_id,
                name=
                    "source_node_id",
            )

        # ==============================================================
        # DETECTOR EVENT ID
        # ==============================================================
        #
        # Detector-event IDs are local software identifiers rather than
        # protocol node IDs.
        #
        # Zero is retained as valid because test/synthetic contexts may
        # use zero as their first local event index.
        # ==============================================================

        if (
            self.detector_event_id
            is not None
        ):

            _require_uint32(
                self.detector_event_id,
                name=
                    "detector_event_id",
                allow_zero=
                    True,
            )

        # ==============================================================
        # SESSION ID
        # ==============================================================
        #
        # Real acquisition sessions are non-zero Protocol-v4 uint32 IDs.
        #
        # None means no acquisition-session context was supplied.
        # ==============================================================

        if (
            self.session_id
            is not None
        ):

            _require_uint32(
                self.session_id,
                name=
                    "session_id",
                allow_zero=
                    False,
            )

        # ==============================================================
        # MODEL AUDIO
        # ==============================================================

        if (
            self.model_audio
            is not None
        ):

            # ----------------------------------------------------------
            # REQUIRE NDARRAY
            # ----------------------------------------------------------
            #
            # The classifier interface deliberately uses ndarray rather
            # than silently converting arbitrary array-like containers.
            #
            # This keeps backend behavior and memory/layout semantics
            # predictable.
            # ----------------------------------------------------------

            if not isinstance(
                self.model_audio,
                np.ndarray,
            ):

                raise TypeError(
                    (
                        "model_audio must be a "
                        "NumPy ndarray or None."
                    )
                )

            audio = (
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
                        "a mono 1-D waveform."
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
                        "cannot be empty."
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
                        "contain numeric samples."
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
                        "real-valued samples."
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
                        "NaN or infinite samples."
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
    Abstract interface implemented by acoustic-classification backends.

    Current implementation
    ----------------------
    HeuristicClassifierBackend:

        requires_features = True
        requires_audio = False


    Future implementations
    ----------------------
    Pretrained waveform model:

        requires_features = False
        requires_audio = True

    Ensemble model:

        may require both


    Architecture
    ------------
    This abstraction keeps:

        EventPipeline
        SQLite persistence
        CLI
        dashboard
        research analytics

    independent from the concrete classifier implementation.
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
        Stable human-readable backend identifier.
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

        Feature-only classifiers leave this False.
        """

        return False

    @property
    def requires_features(
        self,
    ) -> bool:
        """
        Whether handcrafted DSP features are mandatory.

        The current heuristic backend requires features.
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

        Generic structural validation has already been performed by
        ClassificationInput.__post_init__().

        This method enforces backend-specific availability requirements.
        """

        # ==============================================================
        # INPUT TYPE
        # ==============================================================

        if not isinstance(
            classification_input,
            ClassificationInput,
        ):

            raise TypeError(
                (
                    "classification_input must be "
                    "a ClassificationInput instance."
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
                    "model_audio."
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
                    "acoustic features."
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

        Implementations must return ClassificationResult so that the
        surrounding system remains classifier-backend independent.

        Implementations should normally call:

            self.validate_input(classification_input)

        before accessing backend-required input data.
        """

        raise NotImplementedError