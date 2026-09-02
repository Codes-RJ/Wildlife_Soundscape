from __future__ import annotations

from .base import (
    ClassificationInput,
    ClassifierBackend,
)

from .classifier import (
    ClassificationResult,
    HeuristicClassifier,
)


# ======================================================================
# HEURISTIC CLASSIFIER BACKEND
# ======================================================================


class HeuristicClassifierBackend(
    ClassifierBackend
):
    """
    Adapter exposing HeuristicClassifier through the common
    ClassifierBackend interface.

    Architecture
    ------------
    EventPipeline
        ↓
    ClassifierBackend
        ↓
    HeuristicClassifierBackend
        ↓
    HeuristicClassifier

    Input requirements
    ------------------
    DSP features:
        required

    Waveform audio:
        not required

    This backend therefore continues to operate as the transparent,
    feature-based baseline classifier while allowing future backends to
    use waveform audio instead.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        classifier: HeuristicClassifier | None = None,
    ) -> None:
        """
        Create the heuristic classification backend.

        Parameters
        ----------
        classifier
            Optional pre-created HeuristicClassifier instance.

            Supplying one is useful for dependency injection or future
            configuration/testing.

            If omitted, the default classifier is created.
        """

        self._classifier = (
            classifier
            if classifier is not None
            else HeuristicClassifier()
        )

    # ==================================================================
    # BACKEND IDENTITY
    # ==================================================================

    @property
    def name(
        self,
    ) -> str:
        """
        Stable backend identifier.
        """

        return (
            "heuristic_backend"
        )

    @property
    def version(
        self,
    ) -> str:
        """
        Adapter-interface version.

        The underlying HeuristicClassifier stores its own classifier
        identity/version inside ClassificationResult.
        """

        return (
            "1.0"
        )

    # ==================================================================
    # CAPABILITIES
    # ==================================================================

    @property
    def requires_audio(
        self,
    ) -> bool:
        """
        The heuristic backend does not require waveform audio.
        """

        return False

    @property
    def requires_features(
        self,
    ) -> bool:
        """
        The heuristic backend requires handcrafted acoustic features.
        """

        return True

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:
        """
        Classify one acoustic event using handcrafted DSP features.

        The generic ClassificationInput now allows features=None because
        future waveform-only classifiers may not need them.

        This particular backend does require features, so the requirement
        is enforced here through validate_input().
        """

        # --------------------------------------------------------------
        # BACKEND REQUIREMENT VALIDATION
        # --------------------------------------------------------------

        self.validate_input(
            classification_input
        )

        # --------------------------------------------------------------
        # TYPE NARROWING / DEFENSIVE CHECK
        # --------------------------------------------------------------
        #
        # validate_input() already guarantees this at runtime.
        #
        # The explicit local check is retained because static type
        # checkers cannot necessarily infer that a method call proved an
        # Optional field is now non-None.
        # --------------------------------------------------------------

        features = (
            classification_input.features
        )

        if (
            features
            is None
        ):

            raise ValueError(
                (
                    "heuristic classifier backend "
                    "requires acoustic features"
                )
            )

        # --------------------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------------------

        return (
            self._classifier.classify(
                features
            )
        )

    # ==================================================================
    # UNDERLYING CLASSIFIER ACCESS
    # ==================================================================

    @property
    def classifier(
        self,
    ) -> HeuristicClassifier:
        """
        Expose the wrapped HeuristicClassifier for diagnostics.

        Normal application code should interact with this object through
        the ClassifierBackend interface rather than depending directly on
        the underlying classifier.
        """

        return (
            self._classifier
        )