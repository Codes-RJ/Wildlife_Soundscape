"""
Ensemble acoustic-classification backend.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Combine predictions from multiple ClassifierBackend implementations into
one broad acoustic-class decision.

The initial ensemble is:

    heuristic classifier
        +
    BirdNET classifier

The implementation remains backend-independent and can later combine:

    trained project models
    additional bioacoustic classifiers
    specialist taxonomic models


Fusion ontology
---------------
Ensemble fusion occurs only over the project's common broad ontology:

    bird
    insect
    amphibian
    mammal
    noise
    unknown

Specialist scores such as:

    species:Corvus splendens_House Crow

remain available for traceability but do not directly participate in
broad-class voting.


Confidence-preserving fusion
----------------------------
Backends are NOT blindly normalized so that their score vectors sum to
one.

Doing that would incorrectly transform specialist evidence such as:

    BirdNET bird score = 0.80

into:

    bird = 1.00

simply because BirdNET exposes no competing non-bird species classes.

Instead, each backend's broad score vector is calibrated to its declared:

    ClassificationResult.confidence

while preserving the relative relationships between any broad scores it
provides.

For example:

    raw:
        bird = 0.80
        unknown = 0.00

    result confidence:
        0.80

remains:

        bird = 0.80
        unknown = 0.00


Abstention
----------
A backend may legitimately return:

    label = UNKNOWN
    confidence = 0
    all broad scores = 0

This means:

    "this backend has no usable evidence"

rather than:

    "this backend is 100% certain the class is UNKNOWN"

Such an output is therefore treated as an ensemble abstention.

An abstaining backend:

    remains recorded for traceability
    does not add voting weight
    does not create UNKNOWN evidence


Failure policy
--------------
A member may fail independently because of:

    unavailable optional model runtime
    missing member-specific input
    model initialization failure
    inference failure
    malformed backend output

When:

    continue_on_member_error = True

remaining successful backends can continue.

Importantly, ensemble-level input validation does NOT reject the complete
event merely because one member-specific input is unavailable.

Each backend validates its own requirements.

This makes graceful degradation actually work.


Success vs informative evidence
-------------------------------
These are distinct concepts.

Successful member
    Backend executed and returned ClassificationResult.

Informative member
    Returned usable positive broad-class evidence.

A successful backend may still abstain.

`min_successful_members` refers to execution success.

Only informative members contribute voting weight.


Research traceability
---------------------
Original member scores are retained using stable configured member
indices:

    member:1:heuristic:bird
    member:2:birdnet:bird
    member:2:birdnet:species:...

Member numbering therefore does not change if another member fails.


Scientific interpretation
-------------------------
Ensemble agreement represents combined automated classifier evidence.

It does not constitute:

    biological ground truth
    confirmed species presence
    confirmed animal identity

Species-level predictions require independent verification before being
used as verified ecological observations.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from dataclasses import (
    dataclass,
)

from numbers import (
    Real,
)

from typing import (
    Iterable,
)


# ======================================================================
# CLASSIFICATION CONTRACT
# ======================================================================


from .base import (
    ClassificationInput,
    ClassifierBackend,
)


from .classifier import (
    AcousticClass,
    ClassificationResult,
)


# ======================================================================
# CONSTANTS
# ======================================================================


ENSEMBLE_VERSION = "1.1"


DEFAULT_MEMBER_WEIGHT = 1.0


DEFAULT_MIN_SUCCESSFUL_MEMBERS = 1


# ======================================================================
# NUMERIC VALIDATION
# ======================================================================


def _finite_non_negative(
    value: object,
    *,
    name: str,
) -> float:
    """
    Normalize one finite non-negative numerical value.
    """

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        Real,
    ):
        raise TypeError(f"{name} must be numeric.")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    if result < 0.0:
        raise ValueError(f"{name} cannot be negative.")

    return result


# ======================================================================
# POSITIVE NUMBER
# ======================================================================


def _finite_positive(
    value: object,
    *,
    name: str,
) -> float:
    """
    Normalize one finite strictly-positive numerical value.
    """

    result = _finite_non_negative(
        value,
        name=name,
    )

    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


# ======================================================================
# CLAMP TO CLASSIFICATION SCORE RANGE
# ======================================================================


def _clamp01(
    value: float,
) -> float:
    """
    Clamp finite numerical value into [0, 1].
    """

    if not math.isfinite(value):
        raise ValueError("Classification score must be finite.")

    return float(
        max(
            0.0,
            min(
                1.0,
                value,
            ),
        )
    )


# ======================================================================
# OPTIONAL SCORE NORMALIZATION
# ======================================================================


def _score_or_none(
    value: object,
) -> float | None:
    """
    Convert a candidate classification score into [0, 1].

    Invalid values return None rather than compromising the complete
    ensemble result.
    """

    try:
        result = _finite_non_negative(
            value,
            name="classification score",
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if result > 1.0:
        return None

    return result


# ======================================================================
# ENSEMBLE MEMBER
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EnsembleMember:
    """
    One classifier participating in ensemble fusion.

    backend
        Operational ClassifierBackend implementation.

    weight
        Relative voting authority when the backend provides informative
        evidence.

    label
        Optional human-readable member identifier.

        When omitted:

            backend.name

        is used.
    """

    backend: ClassifierBackend

    weight: float = DEFAULT_MEMBER_WEIGHT

    label: str | None = None

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        # ==============================================================
        # BACKEND
        # ==============================================================

        if not isinstance(
            self.backend,
            ClassifierBackend,
        ):
            raise TypeError(("backend must be a ClassifierBackend instance."))

        # ==============================================================
        # WEIGHT
        # ==============================================================

        normalized_weight = _finite_positive(
            self.weight,
            name="Ensemble member weight",
        )

        object.__setattr__(
            self,
            "weight",
            normalized_weight,
        )

        # ==============================================================
        # LABEL
        # ==============================================================

        if self.label is not None:
            if not isinstance(
                self.label,
                str,
            ):
                raise TypeError(("Ensemble member label must be a string or None."))

            normalized_label = self.label.strip()

            if not (normalized_label):
                raise ValueError(("Ensemble member label cannot be empty."))

            object.__setattr__(
                self,
                "label",
                normalized_label,
            )

    # ==================================================================
    # IDENTIFIER
    # ==================================================================

    @property
    def identifier(
        self,
    ) -> str:
        """
        Stable human-readable member identifier.
        """

        if self.label is not None:
            return self.label

        return str(self.backend.name)


# ======================================================================
# MEMBER EXECUTION RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EnsembleMemberResult:
    """
    Successful backend result together with stable ensemble metadata.
    """

    member_index: int

    member: EnsembleMember

    result: ClassificationResult

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        if isinstance(
            self.member_index,
            bool,
        ) or not isinstance(
            self.member_index,
            int,
        ):
            raise TypeError(("member_index must be an integer."))

        if self.member_index <= 0:
            raise ValueError(("member_index must be greater than zero."))

        if not isinstance(
            self.member,
            EnsembleMember,
        ):
            raise TypeError(("member must be an EnsembleMember."))

        if not isinstance(
            self.result,
            ClassificationResult,
        ):
            raise TypeError(("result must be a ClassificationResult."))


# ======================================================================
# BROAD SCORE EXTRACTION
# ======================================================================


def _broad_scores(
    result: ClassificationResult,
) -> dict[
    AcousticClass,
    float,
]:
    """
    Convert one backend result into confidence-preserving broad evidence.

    Specialist scores such as:

        species:...
        birdnet:...

    are excluded from fusion.


    Calibration strategy
    --------------------
    If the backend supplies explicit broad scores and the raw score for
    its declared primary label is positive:

        scale =
            result.confidence / raw_primary_score

    The whole broad score vector is scaled using that factor.

    This makes the declared primary score equal the backend's externally
    reported confidence while preserving its internal class-score ratios.


    Example
    -------
    Backend result:

        label      = bird
        confidence = 0.60

        raw scores:
            bird   = 0.75
            insect = 0.30
            noise  = 0.10

    Scaled evidence:

        bird   = 0.60
        insect = 0.24
        noise  = 0.08


    Abstention
    ----------
    UNKNOWN with zero confidence and no positive broad evidence returns:

        {}

    meaning:

        no ensemble vote
    """

    if not isinstance(
        result,
        ClassificationResult,
    ):
        raise TypeError(("result must be a ClassificationResult."))

    if not isinstance(
        result.label,
        AcousticClass,
    ):
        raise TypeError(("ClassificationResult.label must be an AcousticClass."))

    # ==================================================================
    # DECLARED CONFIDENCE
    # ==================================================================

    confidence = _score_or_none(result.confidence)

    if confidence is None:
        raise ValueError(
            ("ClassificationResult.confidence must be finite and lie in [0, 1].")
        )

    # ==================================================================
    # EXPLICIT BROAD SCORES
    # ==================================================================

    explicit_scores: dict[
        AcousticClass,
        float,
    ] = {}

    for acoustic_class in AcousticClass:
        raw_value = result.scores.get(acoustic_class.value)

        if raw_value is None:
            continue

        score = _score_or_none(raw_value)

        if score is None:
            continue

        explicit_scores[acoustic_class] = score

    # ==================================================================
    # ZERO-CONFIDENCE ABSTENTION
    # ==================================================================
    #
    # The most important example is BirdNET:
    #
    #     UNKNOWN
    #     confidence=0
    #     bird=0
    #     unknown=0
    #
    # That is not positive UNKNOWN evidence.
    # ==================================================================

    if confidence <= 0.0 and not any(score > 0.0 for score in explicit_scores.values()):
        return {}

    # ==================================================================
    # PRIMARY RAW SCORE
    # ==================================================================

    primary_raw_score = explicit_scores.get(result.label)

    # ==================================================================
    # SCALE EXPLICIT VECTOR TO DECLARED CONFIDENCE
    # ==================================================================

    if primary_raw_score is not None and primary_raw_score > 0.0:
        scale = confidence / primary_raw_score

        calibrated: dict[
            AcousticClass,
            float,
        ] = {}

        for acoustic_class, raw_score in explicit_scores.items():
            scaled_score = raw_score * scale

            # ----------------------------------------------------------
            # The declared primary label must remain at least tied for
            # strongest evidence.
            #
            # A backend whose raw scores contradict its own primary
            # result should not cause an alternative score to exceed the
            # declared confidence after adaptation.
            # ----------------------------------------------------------

            if acoustic_class is not result.label:
                scaled_score = min(
                    scaled_score,
                    confidence,
                )

            calibrated[acoustic_class] = _clamp01(scaled_score)

        calibrated[result.label] = confidence

        # --------------------------------------------------------------
        # If everything becomes zero, this remains an abstention.
        # --------------------------------------------------------------

        if not any(score > 0.0 for score in calibrated.values()):
            return {}

        return calibrated

    # ==================================================================
    # FALLBACK TO DECLARED PRIMARY / SECOND RESULT
    # ==================================================================
    #
    # This supports backends that return only their primary result and
    # do not expose broad scores.
    # ==================================================================

    if confidence <= 0.0:
        return {}

    fallback_scores: dict[
        AcousticClass,
        float,
    ] = {
        result.label: confidence,
    }

    second_label = result.second_label

    if second_label is not None:
        if not isinstance(
            second_label,
            AcousticClass,
        ):
            raise TypeError(
                ("ClassificationResult.second_label must be AcousticClass or None.")
            )

        second_confidence = _score_or_none(result.second_confidence)

        if second_confidence is not None and second_confidence > 0.0:
            fallback_scores[second_label] = min(
                second_confidence,
                confidence,
            )

    return fallback_scores


# ======================================================================
# ENSEMBLE BACKEND
# ======================================================================


class EnsembleClassifierBackend(ClassifierBackend):
    """
    Weighted multi-backend broad acoustic classifier.

    Individual members remain ordinary ClassifierBackend objects.

    Therefore this class does not depend on implementation details of:

        HeuristicClassifierBackend
        BirdNETClassifierBackend
        future project-specific models
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        members: Iterable[EnsembleMember | ClassifierBackend],
        *,
        continue_on_member_error: bool = True,
        min_successful_members: int = (DEFAULT_MIN_SUCCESSFUL_MEMBERS),
    ) -> None:

        # ==============================================================
        # NORMALIZE MEMBERS
        # ==============================================================

        normalized_members: list[EnsembleMember] = []

        for member in members:
            if isinstance(
                member,
                EnsembleMember,
            ):
                normalized_members.append(member)

            elif isinstance(
                member,
                ClassifierBackend,
            ):
                normalized_members.append(
                    EnsembleMember(
                        backend=member,
                    )
                )

            else:
                raise TypeError(
                    (
                        "Ensemble members must be "
                        "EnsembleMember or "
                        "ClassifierBackend instances."
                    )
                )

        # ==============================================================
        # REQUIRE ACTUAL ENSEMBLE
        # ==============================================================

        if len(normalized_members) < 2:
            raise ValueError(
                ("EnsembleClassifierBackend requires at least two classifier members.")
            )

        # ==============================================================
        # MEMBER IDENTIFIER UNIQUENESS
        # ==============================================================

        identifiers = [member.identifier for member in normalized_members]

        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                ("Ensemble member identifiers must be unique for traceability.")
            )

        # ==============================================================
        # ERROR POLICY
        # ==============================================================

        if not isinstance(
            continue_on_member_error,
            bool,
        ):
            raise TypeError(("continue_on_member_error must be bool."))

        # ==============================================================
        # SUCCESS MINIMUM
        # ==============================================================

        if isinstance(
            min_successful_members,
            bool,
        ) or not isinstance(
            min_successful_members,
            int,
        ):
            raise TypeError(("min_successful_members must be an integer."))

        if min_successful_members <= 0:
            raise ValueError(("min_successful_members must be greater than zero."))

        if min_successful_members > len(normalized_members):
            raise ValueError(
                ("min_successful_members cannot exceed number of ensemble members.")
            )

        # ==============================================================
        # STATE
        # ==============================================================

        self.members = tuple(normalized_members)

        self.continue_on_member_error = continue_on_member_error

        self.min_successful_members = min_successful_members

    # ==================================================================
    # IDENTITY
    # ==================================================================

    @property
    def name(
        self,
    ) -> str:
        """
        Backend identity used by persistence.
        """

        return "ensemble"

    @property
    def version(
        self,
    ) -> str:
        """
        Ensemble-fusion implementation version.
        """

        return ENSEMBLE_VERSION

    # ==================================================================
    # UPSTREAM DATA REQUIREMENTS
    # ==================================================================

    @property
    def requires_audio(
        self,
    ) -> bool:
        """
        Whether any configured member can use waveform audio.

        This property tells EventPipeline which input products should be
        prepared.

        It is intentionally NOT used to reject the complete ensemble
        event before individual member execution.
        """

        return any(member.backend.requires_audio for member in self.members)

    @property
    def requires_features(
        self,
    ) -> bool:
        """
        Whether any configured member can use handcrafted DSP features.

        Individual availability is validated by each member when it
        executes.
        """

        return any(member.backend.requires_features for member in self.members)

    # ==================================================================
    # RUN MEMBERS
    # ==================================================================

    def _run_members(
        self,
        classification_input: ClassificationInput,
    ) -> tuple[
        tuple[
            EnsembleMemberResult,
            ...,
        ],
        tuple[
            str,
            ...,
        ],
    ]:
        """
        Execute configured classifier members independently.

        Returns
        -------
        successful_results
            Backends that returned valid ClassificationResult objects.

        failure_reasons
            Traceable descriptions of failed members.
        """

        if not isinstance(
            classification_input,
            ClassificationInput,
        ):
            raise TypeError(
                ("classification_input must be a ClassificationInput instance.")
            )

        successful: list[EnsembleMemberResult] = []

        failures: list[str] = []

        for member_index, member in enumerate(
            self.members,
            start=1,
        ):
            backend = member.backend

            try:
                result: object = backend.classify(classification_input)

            except Exception as exc:
                message = (
                    "ensemble member failed "
                    f"| member={member_index} "
                    f"| identifier={member.identifier} "
                    f"| backend={backend.name} "
                    f"| error={type(exc).__name__}: "
                    f"{exc}"
                )

                if not (self.continue_on_member_error):
                    raise RuntimeError(message) from exc

                failures.append(message)

                continue

            if not isinstance(
                result,
                ClassificationResult,
            ):
                message = (
                    "ensemble member returned invalid "
                    "result type "
                    f"| member={member_index} "
                    f"| identifier={member.identifier} "
                    f"| backend={backend.name} "
                    f"| type={type(result).__name__}"
                )

                if not (self.continue_on_member_error):
                    raise RuntimeError(message)

                failures.append(message)

                continue

            successful.append(
                EnsembleMemberResult(
                    member_index=member_index,
                    member=member,
                    result=result,
                )
            )

        # ==============================================================
        # EXECUTION SUCCESS REQUIREMENT
        # ==============================================================

        if len(successful) < self.min_successful_members:
            failure_suffix = (
                "" if not failures else (" Failures: " + " | ".join(failures))
            )

            raise RuntimeError(
                (
                    "Ensemble classification produced only "
                    f"{len(successful)} successful backend(s); "
                    f"{self.min_successful_members} required."
                    f"{failure_suffix}"
                )
            )

        return (
            tuple(successful),
            tuple(failures),
        )

    # ==================================================================
    # CLASSIFY
    # ==================================================================

    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:
        """
        Run confidence-preserving weighted broad-class fusion.

        Member-specific input requirements are evaluated independently so
        one unavailable model input does not unnecessarily block another
        usable member.
        """

        # ==============================================================
        # GENERIC INPUT TYPE
        # ==============================================================
        #
        # Do NOT call:
        #
        #     self.validate_input(...)
        #
        # here.
        #
        # Ensemble.requires_audio / requires_features represent the union
        # of upstream data products that should preferably be prepared.
        #
        # Using the base validation method would require every union input
        # before any member executes and would defeat graceful member
        # degradation.
        # ==============================================================

        if not isinstance(
            classification_input,
            ClassificationInput,
        ):
            raise TypeError(
                ("classification_input must be a ClassificationInput instance.")
            )

        # ==============================================================
        # MEMBER EXECUTION
        # ==============================================================

        (
            successful_results,
            failures,
        ) = self._run_members(classification_input)

        # ==============================================================
        # ACCUMULATED EVIDENCE
        # ==============================================================

        accumulated: dict[
            AcousticClass,
            float,
        ] = {acoustic_class: 0.0 for acoustic_class in AcousticClass}

        informative_weight = 0.0

        informative_results: list[EnsembleMemberResult] = []

        # ==============================================================
        # OUTPUT TRACEABILITY
        # ==============================================================

        output_scores: dict[
            str,
            float,
        ] = {}

        reasons: list[str] = []

        # ==============================================================
        # SUCCESSFUL MEMBER PROCESSING
        # ==============================================================

        for member_result in successful_results:
            member_index = member_result.member_index

            member = member_result.member

            result = member_result.result

            weight = member.weight

            # ==========================================================
            # PRESERVE ORIGINAL MEMBER SCORES
            # ==========================================================

            for score_name, raw_score in result.scores.items():
                numeric_score = _score_or_none(raw_score)

                if numeric_score is None:
                    continue

                namespaced_key = (
                    f"member:{member_index}:{member.identifier}:{score_name}"
                )

                output_scores[namespaced_key] = numeric_score

            # ==========================================================
            # MEMBER EXECUTION TRACE
            # ==========================================================

            reasons.append(
                (
                    "ensemble member "
                    f"{member_index}: "
                    f"identifier={member.identifier}, "
                    f"backend={result.classifier_name}, "
                    f"version={result.classifier_version}, "
                    f"weight={weight:.3f}, "
                    f"label={result.label.value}, "
                    f"confidence={result.confidence:.3f}"
                )
            )

            # ==========================================================
            # PRESERVE MEMBER EXPLANATIONS
            # ==========================================================

            for member_reason in result.reasons:
                reasons.append(
                    (f"ensemble member {member_index} reason: {member_reason}")
                )

            # ==========================================================
            # BROAD EVIDENCE ADAPTATION
            # ==========================================================

            member_scores = _broad_scores(result)

            # ==========================================================
            # ABSTENTION
            # ==========================================================

            if not (member_scores):
                reasons.append(
                    (
                        "ensemble member "
                        f"{member_index} "
                        f"({member.identifier}) abstained "
                        "from broad fusion because it "
                        "provided no positive broad-class "
                        "evidence"
                    )
                )

                continue

            if not any(score > 0.0 for score in member_scores.values()):
                reasons.append(
                    (
                        "ensemble member "
                        f"{member_index} "
                        f"({member.identifier}) supplied "
                        "only zero broad-class evidence "
                        "and was excluded from voting"
                    )
                )

                continue

            # ==========================================================
            # INFORMATIVE VOTING WEIGHT
            # ==========================================================

            informative_weight += weight

            informative_results.append(member_result)

            # ==========================================================
            # BROAD FUSION
            # ==========================================================

            for acoustic_class, score in member_scores.items():
                accumulated[acoustic_class] += weight * score

                output_scores[
                    (
                        "fusion_input:"
                        f"{member_index}:"
                        f"{member.identifier}:"
                        f"{acoustic_class.value}"
                    )
                ] = score

        # ==============================================================
        # MEMBER FAILURE TRACE
        # ==============================================================

        if failures:
            reasons.extend(failures)

        # ==============================================================
        # ALL SUCCESSFUL MEMBERS ABSTAINED
        # ==============================================================

        if informative_weight <= 0.0:
            for acoustic_class in AcousticClass:
                output_scores[acoustic_class.value] = 0.0

            reasons.append(
                (
                    "all successfully executed ensemble "
                    "members abstained or supplied no "
                    "positive broad-class evidence"
                )
            )

            reasons.append(
                (
                    "ensemble therefore returns UNKNOWN "
                    "with zero confidence rather than "
                    "inventing negative evidence"
                )
            )

            return ClassificationResult(
                label=AcousticClass.UNKNOWN,
                confidence=0.0,
                second_label=None,
                second_confidence=None,
                margin=0.0,
                scores=output_scores,
                reasons=tuple(reasons),
                classifier_name=self.name,
                classifier_version=self.version,
            )

        # ==============================================================
        # NORMALIZED WEIGHTED ENSEMBLE SCORES
        # ==============================================================
        #
        # Divide by the sum of weights from informative members.
        #
        # Importantly, the individual backend score vector itself was NOT
        # normalized to sum to one.
        #
        # This preserves each member's confidence scale.
        # ==============================================================

        combined_scores: dict[
            AcousticClass,
            float,
        ] = {}

        for acoustic_class in AcousticClass:
            score = accumulated[acoustic_class] / informative_weight

            score = _clamp01(score)

            combined_scores[acoustic_class] = score

            output_scores[acoustic_class.value] = score

        # ==============================================================
        # RANKING
        # ==============================================================

        ranking = sorted(
            combined_scores.items(),
            key=lambda item: (
                -item[1],
                item[0].value,
            ),
        )

        if not (ranking):
            raise RuntimeError(("Ensemble produced no broad classification scores."))

        (
            top_class,
            top_score,
        ) = ranking[0]

        # ==============================================================
        # DEFENSIVE ZERO-EVIDENCE CHECK
        # ==============================================================

        if top_score <= 0.0:
            reasons.append(("ensemble broad scores collapsed to zero after fusion"))

            return ClassificationResult(
                label=AcousticClass.UNKNOWN,
                confidence=0.0,
                second_label=None,
                second_confidence=None,
                margin=0.0,
                scores=output_scores,
                reasons=tuple(reasons),
                classifier_name=self.name,
                classifier_version=self.version,
            )

        # ==============================================================
        # SECOND BROAD CLASS
        # ==============================================================

        second_class: AcousticClass | None = None

        second_score: float | None = None

        for (
            candidate_class,
            candidate_score,
        ) in ranking[1:]:
            if candidate_score <= 0.0:
                continue

            second_class = candidate_class

            second_score = candidate_score

            break

        # ==============================================================
        # BROAD DECISION MARGIN
        # ==============================================================

        if second_score is None:
            margin = top_score

        else:
            margin = top_score - second_score

        margin = _clamp01(margin)

        # ==============================================================
        # AGREEMENT INFORMATION
        # ==============================================================

        informative_primary_labels = [
            member_result.result.label for member_result in informative_results
        ]

        if len(informative_primary_labels) == 1:
            reasons.append(
                (
                    "final fusion relied on one "
                    "informative ensemble member; "
                    "other members either abstained "
                    "or did not provide usable evidence"
                )
            )

        elif len(set(informative_primary_labels)) == 1:
            reasons.append(
                (
                    "all informative ensemble members "
                    "agreed on broad class "
                    f"'{informative_primary_labels[0].value}'"
                )
            )

        else:
            label_text = ", ".join(
                member_result.result.label.value
                for member_result in informative_results
            )

            reasons.append(
                (
                    "informative ensemble members "
                    "disagreed on primary broad class: "
                    f"{label_text}"
                )
            )

        # ==============================================================
        # FUSION SUMMARY
        # ==============================================================

        reasons.append(
            (
                "ensemble execution summary: "
                f"successful_members="
                f"{len(successful_results)}, "
                f"informative_members="
                f"{len(informative_results)}, "
                f"configured_members="
                f"{len(self.members)}"
            )
        )

        reasons.append(
            (
                "final ensemble label selected from "
                "confidence-preserving weighted "
                "broad-class evidence"
            )
        )

        reasons.append(
            (
                "ensemble agreement represents "
                "automated classifier consensus "
                "and is not biological ground truth"
            )
        )

        # ==============================================================
        # FINAL RESULT
        # ==============================================================

        return ClassificationResult(
            label=top_class,
            confidence=top_score,
            second_label=second_class,
            second_confidence=second_score,
            margin=margin,
            scores=output_scores,
            reasons=tuple(reasons),
            classifier_name=self.name,
            classifier_version=self.version,
        )
