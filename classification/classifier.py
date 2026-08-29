from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from dsp.features import AcousticFeatures


# ======================================================================
# ACOUSTIC CLASS LABELS
# ======================================================================


class AcousticClass(str, Enum):
    """
    Broad acoustic categories used by the baseline classifier.

    Important
    ---------
    These labels represent acoustic-pattern categories.

    They are NOT species identifications.
    """

    BIRD = "bird"

    INSECT = "insect"

    AMPHIBIAN = "amphibian"

    MAMMAL = "mammal"

    NOISE = "noise"

    UNKNOWN = "unknown"


# ======================================================================
# CLASSIFICATION RESULT
# ======================================================================


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """
    Result produced by the baseline acoustic classifier.

    confidence
        Heuristic decision confidence in the selected label.

    second_label
        Next strongest candidate.

    margin
        Difference between the strongest and second strongest
        class scores.

    scores
        Raw broad-class scores in the range [0, 1].

    reasons
        Human-readable explanation of the decision.
    """

    label: AcousticClass

    confidence: float

    second_label: AcousticClass | None

    second_confidence: float

    margin: float

    scores: dict[str, float]

    reasons: tuple[str, ...]

    classifier_name: str = (
        "heuristic_acoustic_classifier"
    )

    classifier_version: str = (
        "1.1"
    )

    @property
    def is_confident(
        self,
    ) -> bool:
        """
        Whether the classification is strong enough to be presented
        as a reasonably confident broad-group prediction.
        """

        return (
            self.label
            != AcousticClass.UNKNOWN
            and self.confidence >= 0.55
            and self.margin >= 0.10
        )


# ======================================================================
# HEURISTIC BASELINE CLASSIFIER
# ======================================================================


class HeuristicClassifier:
    """
    Transparent broad-group acoustic classifier.

    Purpose
    -------
    This is the project's interpretable baseline classifier.

    It uses event-level acoustic descriptors rather than a trained
    neural network.

    Features used
    -------------
    - dominant frequency
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - spectral flatness
    - spectral flux
    - zero-crossing rate
    - event duration
    - SNR

    Output classes
    --------------
    - bird
    - insect
    - amphibian
    - mammal
    - noise
    - unknown

    Scientific limitation
    ---------------------
    The output means:

        "The acoustic pattern is most consistent with this broad group."

    It does NOT mean:

        "The system has biologically confirmed this animal."

    A trained bioacoustic model can replace or complement this
    classifier later without changing the rest of the event pipeline.
    """

    # ==================================================================
    # DECISION THRESHOLDS
    # ==================================================================

    MIN_ACCEPT_CONFIDENCE = 0.48

    MIN_ACCEPT_MARGIN = 0.075

    CONFIDENT_SCORE = 0.55

    CONFIDENT_MARGIN = 0.10

    # ------------------------------------------------------------------
    # SNR
    # ------------------------------------------------------------------

    EXTREME_LOW_SNR_DB = -3.0

    VERY_LOW_SNR_DB = 1.5

    LOW_SNR_DB = 4.0

    MODERATE_SNR_DB = 8.0

    GOOD_SNR_DB = 12.0

    # ------------------------------------------------------------------
    # Noise characteristics
    # ------------------------------------------------------------------

    HIGH_FLATNESS = 0.60

    VERY_HIGH_FLATNESS = 0.72

    # ==================================================================
    # NUMERIC HELPERS
    # ==================================================================

    @staticmethod
    def _finite(
        value: float,
    ) -> bool:

        try:

            return math.isfinite(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

    @staticmethod
    def _clamp01(
        value: float,
    ) -> float:
        """
        Restrict a numeric value to [0, 1].
        """

        if not math.isfinite(
            float(value)
        ):

            return 0.0

        return float(
            max(
                0.0,
                min(
                    1.0,
                    float(value),
                ),
            )
        )

    # ==================================================================
    # SOFT MEMBERSHIP FUNCTIONS
    # ==================================================================

    @classmethod
    def _range_score(
        cls,
        value: float,
        *,
        low: float,
        high: float,
        softness: float,
    ) -> float:
        """
        Soft membership score for a numeric range.

        value inside range:
            1.0

        value slightly outside range:
            gradually decreases

        value far outside:
            0.0
        """

        if not cls._finite(
            value
        ):

            return 0.0

        value = float(
            value
        )

        if (
            low
            <= value
            <= high
        ):

            return 1.0

        if softness <= 0.0:

            return 0.0

        if value < low:

            return cls._clamp01(
                1.0
                - (
                    low
                    - value
                )
                / softness
            )

        return cls._clamp01(
            1.0
            - (
                value
                - high
            )
            / softness
        )

    @classmethod
    def _low_score(
        cls,
        value: float,
        *,
        ideal_max: float,
        zero_at: float,
    ) -> float:
        """
        Score favoring smaller values.
        """

        if not cls._finite(
            value
        ):

            return 0.0

        value = float(
            value
        )

        if value <= ideal_max:

            return 1.0

        if value >= zero_at:

            return 0.0

        if zero_at <= ideal_max:

            return 0.0

        return cls._clamp01(
            1.0
            - (
                value
                - ideal_max
            )
            / (
                zero_at
                - ideal_max
            )
        )

    @classmethod
    def _high_score(
        cls,
        value: float,
        *,
        zero_below: float,
        ideal_min: float,
    ) -> float:
        """
        Score favoring larger values.
        """

        if not cls._finite(
            value
        ):

            return 0.0

        value = float(
            value
        )

        if value >= ideal_min:

            return 1.0

        if value <= zero_below:

            return 0.0

        if ideal_min <= zero_below:

            return 0.0

        return cls._clamp01(
            (
                value
                - zero_below
            )
            / (
                ideal_min
                - zero_below
            )
        )

    @classmethod
    def _target_score(
        cls,
        value: float,
        *,
        center: float,
        half_width: float,
    ) -> float:
        """
        Triangular soft-membership function centered around a target.

        Useful when a class is more characteristic of a specific
        frequency region rather than a very broad interval.
        """

        if (
            not cls._finite(
                value
            )
            or half_width <= 0.0
        ):

            return 0.0

        distance = abs(
            float(value)
            - center
        )

        return cls._clamp01(
            1.0
            - distance
            / half_width
        )

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    @classmethod
    def _feature_quality_valid(
        cls,
        features: AcousticFeatures,
    ) -> bool:
        """
        Reject invalid, empty or non-finite feature vectors.
        """

        required = (
            features.duration_s,

            features.rms,

            features.peak_amplitude,

            features.crest_factor,

            features.zero_crossing_rate,

            features.dominant_frequency_hz,

            features.spectral_centroid_hz,

            features.spectral_bandwidth_hz,

            features.spectral_rolloff_hz,

            features.spectral_flatness,

            features.spectral_flux,
        )

        if any(
            not cls._finite(
                value
            )

            for value
            in required
        ):

            return False

        if (
            features.duration_s
            <= 0.0
        ):

            return False

        if (
            features.rms
            <= 0.0
        ):

            return False

        if (
            features.peak_amplitude
            <= 0.0
        ):

            return False

        if (
            features.dominant_frequency_hz
            < 0.0
        ):

            return False

        if (
            features.spectral_centroid_hz
            < 0.0
        ):

            return False

        if not (
            0.0
            <= features.spectral_flatness
            <= 1.0
        ):

            return False

        return True

    # ==================================================================
    # SNR QUALITY
    # ==================================================================

    def _snr_quality_factor(
        self,
        snr_db: float | None,
    ) -> float:
        """
        Convert SNR into a reliability multiplier.

        Missing SNR does not automatically invalidate an event because
        noise estimation may occasionally be unavailable.
        """

        if snr_db is None:

            return 0.90

        if not self._finite(
            snr_db
        ):

            return 0.75

        snr = float(
            snr_db
        )

        if (
            snr
            <= self.EXTREME_LOW_SNR_DB
        ):

            return 0.25

        if (
            snr
            <= self.VERY_LOW_SNR_DB
        ):

            return 0.45

        if (
            snr
            <= self.LOW_SNR_DB
        ):

            return 0.65

        if (
            snr
            <= self.MODERATE_SNR_DB
        ):

            return 0.82

        if (
            snr
            <= self.GOOD_SNR_DB
        ):

            return 0.93

        return 1.0

    # ==================================================================
    # BIRD
    # ==================================================================

    def _score_bird(
        self,
        f: AcousticFeatures,
    ) -> tuple[
        float,
        list[str],
    ]:

        reasons: list[str] = []

        # --------------------------------------------------------------
        # Bird calls commonly occupy mid/high audible frequency bands,
        # but frequency alone is not sufficiently discriminative.
        # --------------------------------------------------------------

        dominant = self._range_score(
            f.dominant_frequency_hz,
            low=900.0,
            high=9500.0,
            softness=1800.0,
        )

        centroid = self._range_score(
            f.spectral_centroid_hz,
            low=1500.0,
            high=8500.0,
            softness=1800.0,
        )

        tonal = self._low_score(
            f.spectral_flatness,
            ideal_max=0.20,
            zero_at=0.62,
        )

        flux = self._high_score(
            f.spectral_flux,
            zero_below=0.001,
            ideal_min=0.018,
        )

        zcr = self._range_score(
            f.zero_crossing_rate,
            low=0.025,
            high=0.38,
            softness=0.12,
        )

        bandwidth = self._range_score(
            f.spectral_bandwidth_hz,
            low=250.0,
            high=6000.0,
            softness=1800.0,
        )

        duration = self._range_score(
            f.duration_s,
            low=0.04,
            high=5.0,
            softness=2.0,
        )

        rolloff = self._range_score(
            f.spectral_rolloff_hz,
            low=1500.0,
            high=14_000.0,
            softness=2500.0,
        )

        score = (
            0.20
            * dominant

            + 0.17
            * centroid

            + 0.18
            * tonal

            + 0.18
            * flux

            + 0.08
            * zcr

            + 0.07
            * bandwidth

            + 0.05
            * duration

            + 0.07
            * rolloff
        )

        # --------------------------------------------------------------
        # Distinguishing evidence
        # --------------------------------------------------------------

        if (
            dominant >= 0.85
            and centroid >= 0.75
        ):

            reasons.append(
                "dominant and centroid frequencies are consistent "
                "with a mid-to-high frequency vocalization"
            )

        if tonal >= 0.75:

            reasons.append(
                "spectrum contains strong tonal structure"
            )

        if flux >= 0.75:

            reasons.append(
                "frequency content changes strongly over time"
            )

        if (
            tonal >= 0.70
            and flux >= 0.70
        ):

            score += 0.07

            reasons.append(
                "combined tonal and time-varying structure supports "
                "a call/song-like acoustic pattern"
            )

        # Very low-frequency dominance weakens bird evidence.
        if (
            f.dominant_frequency_hz
            < 500.0
        ):

            score *= 0.72

        return (
            self._clamp01(
                score
            ),
            reasons,
        )

    # ==================================================================
    # INSECT
    # ==================================================================

    def _score_insect(
        self,
        f: AcousticFeatures,
    ) -> tuple[
        float,
        list[str],
    ]:

        reasons: list[str] = []

        dominant = self._range_score(
            f.dominant_frequency_hz,
            low=2800.0,
            high=15_800.0,
            softness=2000.0,
        )

        centroid = self._high_score(
            f.spectral_centroid_hz,
            zero_below=2200.0,
            ideal_min=5000.0,
        )

        zcr = self._high_score(
            f.zero_crossing_rate,
            zero_below=0.06,
            ideal_min=0.20,
        )

        rolloff = self._high_score(
            f.spectral_rolloff_hz,
            zero_below=3500.0,
            ideal_min=8000.0,
        )

        duration = self._range_score(
            f.duration_s,
            low=0.10,
            high=12.0,
            softness=2.5,
        )

        bandwidth = self._range_score(
            f.spectral_bandwidth_hz,
            low=400.0,
            high=7000.0,
            softness=1700.0,
        )

        flatness = self._range_score(
            f.spectral_flatness,
            low=0.03,
            high=0.48,
            softness=0.20,
        )

        score = (
            0.24
            * dominant

            + 0.22
            * centroid

            + 0.18
            * zcr

            + 0.12
            * rolloff

            + 0.08
            * duration

            + 0.08
            * bandwidth

            + 0.08
            * flatness
        )

        if (
            dominant >= 0.85
        ):

            reasons.append(
                "dominant acoustic energy occurs at relatively "
                "high frequency"
            )

        if (
            centroid >= 0.80
        ):

            reasons.append(
                "spectral centroid is strongly weighted toward "
                "high frequencies"
            )

        if (
            zcr >= 0.80
        ):

            reasons.append(
                "high zero-crossing activity supports a rapid "
                "high-frequency acoustic pattern"
            )

        if (
            centroid >= 0.75
            and zcr >= 0.75
        ):

            score += 0.06

        # Lower-frequency events should not easily become insects simply
        # because another feature overlaps.
        if (
            f.spectral_centroid_hz
            < 1800.0
        ):

            score *= 0.65

        return (
            self._clamp01(
                score
            ),
            reasons,
        )

    # ==================================================================
    # AMPHIBIAN
    # ==================================================================

    def _score_amphibian(
        self,
        f: AcousticFeatures,
    ) -> tuple[
        float,
        list[str],
    ]:

        reasons: list[str] = []

        dominant = self._range_score(
            f.dominant_frequency_hz,
            low=180.0,
            high=3800.0,
            softness=900.0,
        )

        centroid = self._range_score(
            f.spectral_centroid_hz,
            low=250.0,
            high=3800.0,
            softness=1000.0,
        )

        low_mid_preference = self._target_score(
            f.spectral_centroid_hz,
            center=1750.0,
            half_width=2300.0,
        )

        tonal = self._low_score(
            f.spectral_flatness,
            ideal_max=0.28,
            zero_at=0.70,
        )

        zcr = self._range_score(
            f.zero_crossing_rate,
            low=0.006,
            high=0.22,
            softness=0.09,
        )

        duration = self._range_score(
            f.duration_s,
            low=0.08,
            high=7.0,
            softness=2.0,
        )

        flux = self._range_score(
            f.spectral_flux,
            low=0.001,
            high=0.055,
            softness=0.030,
        )

        score = (
            0.24
            * dominant

            + 0.20
            * centroid

            + 0.13
            * low_mid_preference

            + 0.17
            * tonal

            + 0.10
            * zcr

            + 0.09
            * duration

            + 0.07
            * flux
        )

        if (
            dominant >= 0.85
            and centroid >= 0.75
        ):

            reasons.append(
                "energy is concentrated in a low-to-mid acoustic band"
            )

        if tonal >= 0.75:

            reasons.append(
                "signal contains tonal or harmonic structure"
            )

        if (
            low_mid_preference
            >= 0.75
        ):

            reasons.append(
                "spectral center is consistent with a lower-frequency "
                "calling pattern"
            )

        # Strong high-frequency spectral center weakens the amphibian
        # interpretation.
        if (
            f.spectral_centroid_hz
            > 5000.0
        ):

            score *= 0.60

        return (
            self._clamp01(
                score
            ),
            reasons,
        )

    # ==================================================================
    # MAMMAL
    # ==================================================================

    def _score_mammal(
        self,
        f: AcousticFeatures,
    ) -> tuple[
        float,
        list[str],
    ]:

        reasons: list[str] = []

        dominant = self._range_score(
            f.dominant_frequency_hz,
            low=70.0,
            high=3000.0,
            softness=900.0,
        )

        centroid = self._range_score(
            f.spectral_centroid_hz,
            low=100.0,
            high=3800.0,
            softness=1200.0,
        )

        low_frequency_preference = self._target_score(
            f.spectral_centroid_hz,
            center=1100.0,
            half_width=2300.0,
        )

        zcr = self._low_score(
            f.zero_crossing_rate,
            ideal_max=0.12,
            zero_at=0.40,
        )

        rolloff = self._range_score(
            f.spectral_rolloff_hz,
            low=200.0,
            high=6500.0,
            softness=1600.0,
        )

        duration = self._range_score(
            f.duration_s,
            low=0.04,
            high=8.0,
            softness=3.0,
        )

        bandwidth = self._range_score(
            f.spectral_bandwidth_hz,
            low=150.0,
            high=4500.0,
            softness=1500.0,
        )

        score = (
            0.24
            * dominant

            + 0.19
            * centroid

            + 0.15
            * low_frequency_preference

            + 0.16
            * zcr

            + 0.10
            * rolloff

            + 0.08
            * duration

            + 0.08
            * bandwidth
        )

        if (
            dominant >= 0.85
        ):

            reasons.append(
                "dominant energy lies in a relatively low-frequency "
                "acoustic region"
            )

        if (
            low_frequency_preference
            >= 0.75
        ):

            reasons.append(
                "spectral center is weighted toward lower frequencies"
            )

        if (
            zcr >= 0.80
        ):

            reasons.append(
                "zero-crossing activity is comparatively low"
            )

        if (
            f.spectral_centroid_hz
            > 5500.0
        ):

            score *= 0.55

        return (
            self._clamp01(
                score
            ),
            reasons,
        )

    # ==================================================================
    # NOISE / NON-BIOLOGICAL ACOUSTIC PATTERN
    # ==================================================================

    def _score_noise(
        self,
        f: AcousticFeatures,
    ) -> tuple[
        float,
        list[str],
    ]:

        reasons: list[str] = []

        flatness = self._high_score(
            f.spectral_flatness,
            zero_below=0.25,
            ideal_min=0.68,
        )

        bandwidth = self._high_score(
            f.spectral_bandwidth_hz,
            zero_below=2500.0,
            ideal_min=7000.0,
        )

        low_tonality = self._high_score(
            f.spectral_flatness,
            zero_below=0.38,
            ideal_min=0.72,
        )

        low_snr = 0.0

        if (
            f.snr_db is not None
            and self._finite(
                f.snr_db
            )
        ):

            snr = float(
                f.snr_db
            )

            if (
                snr
                <= self.VERY_LOW_SNR_DB
            ):

                low_snr = 1.0

            elif (
                snr
                < self.MODERATE_SNR_DB
            ):

                low_snr = self._clamp01(
                    1.0
                    - (
                        snr
                        - self.VERY_LOW_SNR_DB
                    )
                    / (
                        self.MODERATE_SNR_DB
                        - self.VERY_LOW_SNR_DB
                    )
                )

        score = (
            0.42
            * flatness

            + 0.22
            * bandwidth

            + 0.16
            * low_tonality

            + 0.20
            * low_snr
        )

        if (
            f.spectral_flatness
            >= self.HIGH_FLATNESS
        ):

            reasons.append(
                "spectrum is comparatively flat and noise-like"
            )

        if (
            bandwidth >= 0.80
        ):

            reasons.append(
                "energy is distributed across a broad frequency range"
            )

        if (
            low_snr >= 0.75
        ):

            reasons.append(
                "event has poor signal-to-noise separation"
            )

        # Strong broadband + flat-spectrum evidence.
        if (
            f.spectral_flatness
            >= self.VERY_HIGH_FLATNESS
            and bandwidth >= 0.70
        ):

            score += 0.10

        return (
            self._clamp01(
                score
            ),
            reasons,
        )

    # ==================================================================
    # SCORE ALL CLASSES
    # ==================================================================

    def _calculate_scores(
        self,
        features: AcousticFeatures,
    ) -> tuple[
        dict[AcousticClass, float],
        dict[AcousticClass, list[str]],
    ]:

        (
            bird_score,
            bird_reasons,
        ) = self._score_bird(
            features
        )

        (
            insect_score,
            insect_reasons,
        ) = self._score_insect(
            features
        )

        (
            amphibian_score,
            amphibian_reasons,
        ) = self._score_amphibian(
            features
        )

        (
            mammal_score,
            mammal_reasons,
        ) = self._score_mammal(
            features
        )

        (
            noise_score,
            noise_reasons,
        ) = self._score_noise(
            features
        )

        scores = {
            AcousticClass.BIRD:
                bird_score,

            AcousticClass.INSECT:
                insect_score,

            AcousticClass.AMPHIBIAN:
                amphibian_score,

            AcousticClass.MAMMAL:
                mammal_score,

            AcousticClass.NOISE:
                noise_score,
        }

        reasons = {
            AcousticClass.BIRD:
                bird_reasons,

            AcousticClass.INSECT:
                insect_reasons,

            AcousticClass.AMPHIBIAN:
                amphibian_reasons,

            AcousticClass.MAMMAL:
                mammal_reasons,

            AcousticClass.NOISE:
                noise_reasons,
        }

        return (
            scores,
            reasons,
        )

    # ==================================================================
    # EMPTY SCORE RESULT
    # ==================================================================

    @staticmethod
    def _empty_scores() -> dict[
        str,
        float,
    ]:

        return {
            acoustic_class.value: 0.0

            for acoustic_class
            in AcousticClass
        }

    # ==================================================================
    # CLASSIFY
    # ==================================================================

    def classify(
        self,
        features: AcousticFeatures,
    ) -> ClassificationResult:
        """
        Classify one event-level acoustic feature vector.
        """

        # ==============================================================
        # FEATURE VALIDITY
        # ==============================================================

        if (
            not self._feature_quality_valid(
                features
            )
        ):

            return ClassificationResult(
                label=
                    AcousticClass.UNKNOWN,

                confidence=
                    1.0,

                second_label=
                    None,

                second_confidence=
                    0.0,

                margin=
                    0.0,

                scores=
                    self._empty_scores(),

                reasons=(
                    "acoustic feature vector is incomplete, invalid, "
                    "or contains no usable signal",
                ),
            )

        # ==============================================================
        # EXTREME SNR FAILURE
        # ==============================================================

        if (
            features.snr_db is not None
            and self._finite(
                features.snr_db
            )
            and features.snr_db
            < self.EXTREME_LOW_SNR_DB
        ):

            scores = (
                self._empty_scores()
            )

            scores[
                AcousticClass.NOISE.value
            ] = 0.75

            scores[
                AcousticClass.UNKNOWN.value
            ] = 1.0

            return ClassificationResult(
                label=
                    AcousticClass.UNKNOWN,

                confidence=
                    1.0,

                second_label=
                    AcousticClass.NOISE,

                second_confidence=
                    0.75,

                margin=
                    0.0,

                scores=
                    scores,

                reasons=(
                    "signal-to-noise ratio is too low for reliable "
                    "broad acoustic classification",
                ),
            )

        # ==============================================================
        # CALCULATE RAW CLASS EVIDENCE
        # ==============================================================

        (
            raw_scores,
            reasons_by_class,
        ) = self._calculate_scores(
            features
        )

        # ==============================================================
        # RANK
        # ==============================================================

        ranking = sorted(
            raw_scores.items(),

            key=lambda item:
                item[1],

            reverse=True,
        )

        (
            top_label,
            top_score,
        ) = ranking[0]

        (
            second_label,
            second_score,
        ) = ranking[1]

        raw_margin = (
            float(
                top_score
                - second_score
            )
        )

        # ==============================================================
        # SIGNAL QUALITY
        # ==============================================================

        quality_factor = (
            self._snr_quality_factor(
                features.snr_db
            )
        )

        adjusted_top = (
            self._clamp01(
                top_score
                * quality_factor
            )
        )

        adjusted_second = (
            self._clamp01(
                second_score
                * quality_factor
            )
        )

        # ==============================================================
        # BIOLOGICAL VS NOISE OVERRIDE
        # ==============================================================

        if (
            top_label
            != AcousticClass.NOISE
            and features.spectral_flatness
            >= self.VERY_HIGH_FLATNESS
        ):

            adjusted_top *= 0.72

        # ==============================================================
        # AMBIGUITY
        # ==============================================================

        ambiguous = (
            adjusted_top
            < self.MIN_ACCEPT_CONFIDENCE

            or raw_margin
            < self.MIN_ACCEPT_MARGIN
        )

        score_output: dict[
            str,
            float,
        ] = {
            label.value:
                float(
                    self._clamp01(
                        score
                    )
                )

            for label, score
            in raw_scores.items()
        }

        # ==============================================================
        # UNKNOWN DECISION
        # ==============================================================

        if ambiguous:

            ambiguity_strength = max(
                1.0
                - adjusted_top,

                self._clamp01(
                    (
                        self.MIN_ACCEPT_MARGIN
                        - raw_margin
                    )
                    / self.MIN_ACCEPT_MARGIN
                )
                if (
                    raw_margin
                    < self.MIN_ACCEPT_MARGIN
                )
                else 0.0,
            )

            unknown_confidence = (
                self._clamp01(
                    ambiguity_strength
                )
            )

            score_output[
                AcousticClass.UNKNOWN.value
            ] = unknown_confidence

            reasons: list[str] = [
                (
                    "available acoustic evidence is not sufficiently "
                    "distinct for a reliable broad-group decision"
                )
            ]

            if (
                adjusted_top
                < self.MIN_ACCEPT_CONFIDENCE
            ):

                reasons.append(
                    (
                        f"best candidate score after signal-quality "
                        f"adjustment is only {adjusted_top:.2f}"
                    )
                )

            if (
                raw_margin
                < self.MIN_ACCEPT_MARGIN
            ):

                reasons.append(
                    (
                        f"top candidates overlap strongly: "
                        f"{top_label.value}={top_score:.2f}, "
                        f"{second_label.value}={second_score:.2f}"
                    )
                )

            if (
                features.snr_db
                is not None
                and self._finite(
                    features.snr_db
                )
                and features.snr_db
                < self.LOW_SNR_DB
            ):

                reasons.append(
                    (
                        f"low event SNR "
                        f"({features.snr_db:.1f} dB) reduced "
                        "classification reliability"
                    )
                )

            reasons.append(
                (
                    f"strongest acoustic candidate remains "
                    f"{top_label.value}"
                )
            )

            return ClassificationResult(
                label=
                    AcousticClass.UNKNOWN,

                confidence=
                    unknown_confidence,

                second_label=
                    top_label,

                second_confidence=
                    adjusted_top,

                margin=
                    raw_margin,

                scores=
                    score_output,

                reasons=
                    tuple(
                        reasons
                    ),
            )

        # ==============================================================
        # ACCEPTED CLASSIFICATION
        # ==============================================================

        score_output[
            AcousticClass.UNKNOWN.value
        ] = 0.0

        accepted_reasons = list(
            reasons_by_class[
                top_label
            ]
        )

        # --------------------------------------------------------------
        # Generic useful explanation
        # --------------------------------------------------------------

        accepted_reasons.append(
            (
                f"{top_label.value} produced the highest broad-group "
                f"acoustic score ({top_score:.2f})"
            )
        )

        accepted_reasons.append(
            (
                f"decision margin over {second_label.value} "
                f"is {raw_margin:.2f}"
            )
        )

        if (
            features.snr_db
            is not None
            and self._finite(
                features.snr_db
            )
        ):

            accepted_reasons.append(
                (
                    f"event SNR={features.snr_db:.1f} dB"
                )
            )

        accepted_reasons.append(
            (
                "classification is a broad acoustic-pattern estimate, "
                "not species identification"
            )
        )

        return ClassificationResult(
            label=
                top_label,

            confidence=
                adjusted_top,

            second_label=
                second_label,

            second_confidence=
                adjusted_second,

            margin=
                raw_margin,

            scores=
                score_output,

            reasons=
                tuple(
                    accepted_reasons
                ),
        )