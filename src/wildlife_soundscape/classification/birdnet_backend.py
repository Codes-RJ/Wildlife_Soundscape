"""
Optional BirdNET acoustic-classification backend.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Integrate BirdNET species-level acoustic inference into the project's
backend-independent classification architecture.

The core EventPipeline depends only on:

    ClassifierBackend
        ↓
    ClassificationResult

This module therefore adapts BirdNET predictions into the existing
classification contract without introducing BirdNET dependencies into:

    event detection
    DSP
    localization
    database persistence
    analytics
    dashboard code


Current BirdNET integration
---------------------------
The preferred implementation uses the official ``birdnet`` Python
package and the BirdNET V3 acoustic model.

Default model:

    BirdNET acoustic V3.0

Default runtime backend:

    ONNX

Default precision:

    FP32


BirdNET model audio
-------------------
The project acquires audio at:

    48 kHz

BirdNET V3 internally operates using its model-specific sampling
requirements and automatically resamples supported audio input.

The acquisition configuration therefore remains independent from the
BirdNET model's internal sample rate.


Classification-contract limitation
----------------------------------
The project-wide ClassificationResult represents broad acoustic classes:

    bird
    insect
    amphibian
    mammal
    noise
    unknown

BirdNET produces species-level bird detections.

The adapter therefore operates as a specialist classifier:

    positive BirdNET detection
        ->
    BIRD

    no accepted BirdNET detection
        ->
    UNKNOWN / abstention


Species evidence
----------------
Species-level predictions are preserved inside the scores dictionary
using namespaced keys:

    species:<BirdNET taxonomy label>

Example:

    {
        "bird": 0.91,
        "unknown": 0.0,
        "species:Corvus splendens_House Crow": 0.91,
        "species:Psittacula krameri_Rose-ringed Parakeet": 0.34,
        "birdnet:species_margin": 0.57,
    }


Broad margin vs species margin
------------------------------
These values must not be confused.

ClassificationResult.margin represents the broad-class decision margin.

For an accepted BirdNET detection:

    BIRD score
        =
    strongest accepted BirdNET species confidence

    UNKNOWN score
        =
    0

Therefore:

    broad margin
        =
    BIRD score - UNKNOWN score


The BirdNET top-species-vs-second-species margin is retained separately
as:

    scores["birdnet:species_margin"]

and in human-readable reasons.


No-detection interpretation
---------------------------
Failure to detect a bird above the configured threshold does NOT imply:

    100% certainty that the sound is not biological
    100% confidence in another acoustic class

BirdNET is primarily a bird-species recognition model.

Therefore no accepted species prediction is represented conservatively
as:

    label = UNKNOWN
    confidence = 0
    margin = 0

This represents abstention rather than strong negative evidence.


Scientific interpretation
-------------------------
BirdNET predictions are automated acoustic-model outputs.

A high model score must not automatically be presented as verified
species presence.

Research reporting should distinguish:

    automated prediction
    manually verified observation
    ground-truth reference label


Dependency policy
-----------------
BirdNET is OPTIONAL.

Importing this module does not import BirdNET itself.

The external package is loaded lazily only when the backend first needs
to load its model.

This allows the remainder of the Wildlife Soundscape system to operate
without BirdNET installed.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import csv
import importlib
import importlib.metadata
import math
import tempfile
import wave

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Iterable,
)


# ======================================================================
# THIRD PARTY
# ======================================================================


import numpy as np


# ======================================================================
# PROJECT CLASSIFICATION CONTRACT
# ======================================================================


from .base import (
    ClassificationInput,
    ClassifierBackend,
)


from .birdnet_context import (
    BirdNETGeoContext,
    BirdNETTaxonomy,
)


from .classifier import (
    AcousticClass,
    ClassificationResult,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_MODEL_VERSION = (
    "3.0"
)


DEFAULT_MODEL_BACKEND = (
    "onnx"
)


DEFAULT_MODEL_PRECISION = (
    "fp32"
)


DEFAULT_MIN_CONFIDENCE = (
    0.20
)


DEFAULT_TOP_K = (
    5
)


# ======================================================================
# BIRDNET SPECIES PREDICTION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BirdNETSpeciesPrediction:
    """
    One aggregated BirdNET species prediction.

    species_name
        BirdNET taxonomy label.

        Common form:

            Scientific name_Common name

    confidence
        Maximum BirdNET confidence observed for this species across the
        model windows overlapping the supplied acoustic event.

    detection_count
        Number of BirdNET output rows associated with this species.
    """

    species_name: str

    confidence: float

    detection_count: int

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        # ==============================================================
        # SPECIES NAME
        # ==============================================================

        if not isinstance(
            self.species_name,
            str,
        ):

            raise TypeError(
                (
                    "species_name must "
                    "be a string."
                )
            )

        species_name = (
            self.species_name
            .strip()
        )

        if not (
            species_name
        ):

            raise ValueError(
                (
                    "species_name cannot "
                    "be empty."
                )
            )

        object.__setattr__(
            self,
            "species_name",
            species_name,
        )

        # ==============================================================
        # CONFIDENCE
        # ==============================================================

        try:

            confidence = float(
                self.confidence
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "confidence must "
                    "be numeric."
                )
            ) from exc

        if not math.isfinite(
            confidence
        ):

            raise ValueError(
                (
                    "confidence must "
                    "be finite."
                )
            )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):

            raise ValueError(
                (
                    "confidence must lie "
                    "between 0 and 1."
                )
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        # ==============================================================
        # DETECTION COUNT
        # ==============================================================

        if (
            isinstance(
                self.detection_count,
                bool,
            )
            or not isinstance(
                self.detection_count,
                int,
            )
        ):

            raise TypeError(
                (
                    "detection_count must "
                    "be an integer."
                )
            )

        if (
            self.detection_count
            <= 0
        ):

            raise ValueError(
                (
                    "detection_count must "
                    "be greater than zero."
                )
            )

    # ==================================================================
    # SCIENTIFIC NAME
    # ==================================================================

    @property
    def scientific_name(
        self,
    ) -> str:
        """
        Extract scientific component from BirdNET taxonomy label.
        """

        if (
            "_"
            in self.species_name
        ):

            return (
                self.species_name
                .split(
                    "_",
                    1,
                )[
                    0
                ]
                .strip()
            )

        return (
            self.species_name
        )

    # ==================================================================
    # COMMON NAME
    # ==================================================================

    @property
    def common_name(
        self,
    ) -> str | None:
        """
        Extract common-name component when present.
        """

        if (
            "_"
            not in self.species_name
        ):

            return (
                None
            )

        common_name = (
            self.species_name
            .split(
                "_",
                1,
            )[
                1
            ]
            .strip()
        )

        return (
            common_name
            if common_name
            else None
        )


# ======================================================================
# BIRDNET BACKEND
# ======================================================================


class BirdNETClassifierBackend(
    ClassifierBackend
):
    """
    Optional BirdNET waveform-classification backend.

    BirdNET performs species inference.

    This adapter converts species-level predictions into the project's
    broad ClassificationResult ontology while preserving species
    evidence in namespaced scores and explanations.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        *,
        model_version: str = DEFAULT_MODEL_VERSION,
        model_backend: str = DEFAULT_MODEL_BACKEND,
        model_precision: str = DEFAULT_MODEL_PRECISION,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        top_k: int = DEFAULT_TOP_K,
        custom_species_list: Path | None = None,
        taxonomy: BirdNETTaxonomy | None = None,
        geo_context: BirdNETGeoContext | None = None,
        geo_model: Any | None = None,
        model: Any | None = None,
    ) -> None:

        # ==============================================================
        # MODEL VERSION
        # ==============================================================

        if not isinstance(
            model_version,
            str,
        ):

            raise TypeError(
                (
                    "model_version must "
                    "be a string."
                )
            )

        model_version = (
            model_version
            .strip()
        )

        if not (
            model_version
        ):

            raise ValueError(
                (
                    "model_version cannot "
                    "be empty."
                )
            )

        # ==============================================================
        # MODEL BACKEND
        # ==============================================================

        if not isinstance(
            model_backend,
            str,
        ):

            raise TypeError(
                (
                    "model_backend must "
                    "be a string."
                )
            )

        model_backend = (
            model_backend
            .strip()
            .lower()
        )

        if not (
            model_backend
        ):

            raise ValueError(
                (
                    "model_backend cannot "
                    "be empty."
                )
            )

        # ==============================================================
        # MODEL PRECISION
        # ==============================================================

        if not isinstance(
            model_precision,
            str,
        ):

            raise TypeError(
                (
                    "model_precision must "
                    "be a string."
                )
            )

        model_precision = (
            model_precision
            .strip()
            .lower()
        )

        if not (
            model_precision
        ):

            raise ValueError(
                (
                    "model_precision cannot "
                    "be empty."
                )
            )

        # ==============================================================
        # MINIMUM CONFIDENCE
        # ==============================================================

        try:

            min_confidence = float(
                min_confidence
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "min_confidence must "
                    "be numeric."
                )
            ) from exc

        if (
            not math.isfinite(
                min_confidence
            )
            or not (
                0.0
                <= min_confidence
                <= 1.0
            )
        ):

            raise ValueError(
                (
                    "min_confidence must lie "
                    "between 0 and 1."
                )
            )

        # ==============================================================
        # TOP K
        # ==============================================================

        if (
            isinstance(
                top_k,
                bool,
            )
            or not isinstance(
                top_k,
                int,
            )
        ):

            raise TypeError(
                (
                    "top_k must "
                    "be an integer."
                )
            )

        if (
            top_k
            <= 0
        ):

            raise ValueError(
                (
                    "top_k must be "
                    "greater than zero."
                )
            )

        # ==============================================================
        # CUSTOM SPECIES LIST
        # ==============================================================

        if (
            custom_species_list
            is not None
        ):

            if not isinstance(
                custom_species_list,
                Path,
            ):

                raise TypeError(
                    (
                        "custom_species_list must "
                        "be pathlib.Path or None."
                    )
                )

            custom_species_list = (
                custom_species_list
                .expanduser()
            )

        # ==============================================================
        # STATE
        # ==============================================================

        self.model_version = (
            model_version
        )

        self.model_backend = (
            model_backend
        )

        self.model_precision = (
            model_precision
        )

        self.min_confidence = (
            min_confidence
        )

        self.top_k = (
            top_k
        )

        self.custom_species_list = (
            custom_species_list
        )

        self.taxonomy = (
            taxonomy
            if taxonomy is not None
            else BirdNETTaxonomy()
        )

        self.geo_context = (
            geo_context
            if geo_context is not None
            else BirdNETGeoContext()
        )

        self._geo_model = (
            geo_model
        )

        self._geo_model_attempted = (
            geo_model is not None
        )

        self._geo_model_error: str | None = (
            None
        )

        self._model = (
            model
        )

        self._birdnet_package_version: (
            str
            | None
        ) = (
            None
        )

    # ==================================================================
    # IDENTITY
    # ==================================================================

    @property
    def name(
        self,
    ) -> str:
        """
        Backend identity used by persistence and logs.
        """

        return (
            "birdnet"
        )

    @property
    def version(
        self,
    ) -> str:
        """
        Human-readable model/runtime version.
        """

        package_version = (
            self._birdnet_package_version
        )

        if (
            package_version
            is None
        ):

            try:

                package_version = (
                    importlib
                    .metadata
                    .version(
                        "birdnet"
                    )
                )

            except (
                importlib
                .metadata
                .PackageNotFoundError
            ):

                package_version = (
                    "not-loaded"
                )

        return (
            f"model-{self.model_version}"
            f"/{self.model_backend}"
            f"/{self.model_precision}"
            f"/birdnet-{package_version}"
        )

    # ==================================================================
    # BACKEND CAPABILITIES
    # ==================================================================

    @property
    def requires_audio(
        self,
    ) -> bool:
        """
        BirdNET requires waveform audio.
        """

        return (
            True
        )

    @property
    def requires_features(
        self,
    ) -> bool:
        """
        BirdNET does not require handcrafted project DSP features.
        """

        return (
            False
        )

    # ==================================================================
    # OPTIONAL DEPENDENCY LOAD
    # ==================================================================

    def _load_birdnet_module(
        self,
    ):
        """
        Import official BirdNET package lazily.
        """

        try:

            birdnet = (
                importlib.import_module(
                    "birdnet"
                )
            )

        except ImportError as exc:

            raise RuntimeError(
                (
                    "BirdNET backend is selected but "
                    "the optional 'birdnet' package "
                    "is not installed."
                )
            ) from exc

        try:

            self._birdnet_package_version = (
                importlib
                .metadata
                .version(
                    "birdnet"
                )
            )

        except (
            importlib
            .metadata
            .PackageNotFoundError
        ):

            self._birdnet_package_version = (
                "unknown"
            )

        return (
            birdnet
        )

    # ==================================================================
    # MODEL LOAD
    # ==================================================================

    def _ensure_model(
        self,
    ):
        """
        Lazily initialize the configured BirdNET acoustic model.

        Default:

            acoustic 3.0
            ONNX
            FP32
        """

        if (
            self._model
            is not None
        ):

            return (
                self._model
            )

        birdnet = (
            self._load_birdnet_module()
        )

        load_function = getattr(
            birdnet,
            "load",
            None,
        )

        if not callable(
            load_function
        ):

            raise RuntimeError(
                (
                    "Installed BirdNET package "
                    "does not expose birdnet.load()."
                )
            )

        try:

            self._model = (
                load_function(
                    "acoustic",
                    self.model_version,
                    self.model_backend,
                    precision=
                        self.model_precision,
                )
            )

        except Exception as exc:

            raise RuntimeError(
                (
                    "Unable to initialize BirdNET "
                    "acoustic model "
                    f"{self.model_version} "
                    f"using backend "
                    f"'{self.model_backend}' "
                    f"and precision "
                    f"'{self.model_precision}'."
                )
            ) from exc

        return (
            self._model
        )

    # ==================================================================
    # OPTIONAL GEOGRAPHIC MODEL LOAD
    # ==================================================================

    def _ensure_geo_model(
        self,
    ):
        """Lazily initialize optional BirdNET geographic context."""

        if not self.geo_context.enabled:
            return None

        if self._geo_model is not None:
            return self._geo_model

        if self._geo_model_attempted:
            return None

        self._geo_model_attempted = True

        try:
            birdnet = self._load_birdnet_module()
            load_function = getattr(birdnet, "load", None)
            if not callable(load_function):
                raise RuntimeError(
                    "Installed BirdNET package does not expose birdnet.load()."
                )

            self._geo_model = load_function(
                "geo",
                self.model_version,
                self.model_backend,
                precision=self.model_precision,
            )
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            self._geo_model_error = detail
            return None

        return self._geo_model

    # ==================================================================
    # MODEL AUDIO NORMALIZATION
    # ==================================================================

    @staticmethod
    def _audio_to_pcm16(
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        Convert backend waveform into mono little-endian PCM16 samples.

        Floating-point model audio is expected approximately within:

            [-1, +1]

        Values outside that interval are clipped defensively.
        """

        waveform = (
            np.asarray(
                audio
            )
        )

        if (
            waveform.ndim
            != 1
        ):

            raise ValueError(
                (
                    "BirdNET model audio "
                    "must be one-dimensional."
                )
            )

        if (
            waveform.size
            == 0
        ):

            raise ValueError(
                (
                    "BirdNET model audio "
                    "cannot be empty."
                )
            )

        if not np.all(
            np.isfinite(
                waveform
            )
        ):

            raise ValueError(
                (
                    "BirdNET model audio contains "
                    "non-finite samples."
                )
            )

        # ==============================================================
        # INTEGER AUDIO
        # ==============================================================

        if np.issubdtype(
            waveform.dtype,
            np.integer,
        ):

            clipped = (
                np.clip(
                    waveform,
                    -32768,
                    32767,
                )
            )

            return (
                np.ascontiguousarray(
                    clipped,
                    dtype=
                        "<i2",
                )
            )

        # ==============================================================
        # FLOAT AUDIO
        # ==============================================================

        floating = (
            np.asarray(
                waveform,
                dtype=
                    np.float64,
            )
        )

        clipped = (
            np.clip(
                floating,
                -1.0,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Positive PCM16 maximum is 32767.
        #
        # -1.0 maps to -32767 here rather than -32768. The one-LSB
        # asymmetry is irrelevant for BirdNET inference and avoids an
        # asymmetric multiplication branch.
        # --------------------------------------------------------------

        pcm = (
            np.rint(
                clipped
                * 32767.0
            )
        )

        return (
            np.ascontiguousarray(
                pcm,
                dtype=
                    "<i2",
            )
        )

    # ==================================================================
    # TEMPORARY WAV
    # ==================================================================

    @staticmethod
    def _write_wave_file(
        path: Path,
        pcm16: np.ndarray,
        *,
        sample_rate: int,
    ) -> None:
        """
        Write temporary mono PCM16 WAV for BirdNET inference.
        """

        if not isinstance(
            path,
            Path,
        ):

            raise TypeError(
                "path must be pathlib.Path."
            )

        if (
            isinstance(
                sample_rate,
                bool,
            )
            or not isinstance(
                sample_rate,
                int,
            )
        ):

            raise TypeError(
                (
                    "sample_rate must "
                    "be an integer."
                )
            )

        if (
            sample_rate
            <= 0
        ):

            raise ValueError(
                (
                    "sample_rate must be "
                    "greater than zero."
                )
            )

        pcm16 = (
            np.asarray(
                pcm16
            )
        )

        if (
            pcm16.ndim
            != 1
        ):

            raise ValueError(
                (
                    "pcm16 must be "
                    "one-dimensional."
                )
            )

        if (
            pcm16.size
            == 0
        ):

            raise ValueError(
                (
                    "pcm16 cannot "
                    "be empty."
                )
            )

        with wave.open(
            str(
                path
            ),
            "wb",
        ) as wav_file:

            wav_file.setnchannels(
                1
            )

            wav_file.setsampwidth(
                2
            )

            wav_file.setframerate(
                sample_rate
            )

            wav_file.writeframes(
                pcm16.astype(
                    "<i2",
                    copy=
                        False,
                )
                .tobytes()
            )

    # ==================================================================
    # CUSTOM SPECIES LIST VALIDATION
    # ==================================================================

    def _custom_species_list_argument(
        self,
    ) -> str | None:
        """
        Validate and return optional BirdNET custom-species-list path.
        """

        if (
            self.custom_species_list
            is None
        ):

            return (
                None
            )

        path = (
            self.custom_species_list
        )

        if not (
            path.exists()
        ):

            raise FileNotFoundError(
                (
                    "BirdNET custom species "
                    "list does not exist: "
                    f"{path}"
                )
            )

        if not (
            path.is_file()
        ):

            raise ValueError(
                (
                    "BirdNET custom species "
                    "list is not a file: "
                    f"{path}"
                )
            )

        return (
            str(
                path
            )
        )

    # ==================================================================
    # RUN MODEL
    # ==================================================================

    def _predict_file(
        self,
        audio_path: Path,
    ):
        """
        Invoke the official BirdNET acoustic prediction API.

        Important
        ---------
        ``top_k`` and ``default_confidence_threshold`` are supplied
        directly to BirdNET.

        Post-processing alone is insufficient because BirdNET may
        discard predictions internally before returning its result.
        """

        model = (
            self._ensure_model()
        )

        predict_function = getattr(
            model,
            "predict",
            None,
        )

        if not callable(
            predict_function
        ):

            raise RuntimeError(
                (
                    "Loaded BirdNET acoustic model "
                    "does not expose predict()."
                )
            )

        prediction_arguments: dict[
            str,
            Any,
        ] = {
            "top_k":
                self.top_k,

            "default_confidence_threshold":
                self.min_confidence,
        }

        custom_species_list = (
            self._custom_species_list_argument()
        )

        if (
            custom_species_list
            is not None
        ):

            prediction_arguments[
                "custom_species_list"
            ] = (
                custom_species_list
            )

        try:

            return (
                predict_function(
                    str(
                        audio_path
                    ),
                    **prediction_arguments,
                )
            )

        except Exception as exc:

            raise RuntimeError(
                (
                    "BirdNET acoustic prediction "
                    "failed."
                )
            ) from exc

    # ==================================================================
    # EXPORT BIRDNET RESULT TO CSV
    # ==================================================================

    @staticmethod
    def _prediction_to_csv(
        prediction_result: Any,
        output_path: Path,
    ) -> None:
        """
        Export BirdNET's tabular prediction result to CSV.
        """

        to_csv = getattr(
            prediction_result,
            "to_csv",
            None,
        )

        if not callable(
            to_csv
        ):

            raise RuntimeError(
                (
                    "BirdNET prediction result "
                    "does not expose to_csv()."
                )
            )

        try:

            to_csv(
                str(
                    output_path
                )
            )

        except TypeError:

            try:

                to_csv(
                    output_path
                )

            except Exception as exc:

                raise RuntimeError(
                    (
                        "Unable to convert BirdNET "
                        "predictions to CSV."
                    )
                ) from exc

        except Exception as exc:

            raise RuntimeError(
                (
                    "Unable to convert BirdNET "
                    "predictions to CSV."
                )
            ) from exc

        if not (
            output_path.exists()
        ):

            raise RuntimeError(
                (
                    "BirdNET prediction CSV "
                    "was not created."
                )
            )

        if not (
            output_path.is_file()
        ):

            raise RuntimeError(
                (
                    "BirdNET prediction CSV path "
                    "is not a file."
                )
            )

    # ==================================================================
    # PARSE BIRDNET CSV
    # ==================================================================

    @staticmethod
    def _read_prediction_rows(
        csv_path: Path,
    ) -> list[
        dict[
            str,
            str,
        ]
    ]:
        """
        Read BirdNET tabular prediction output.
        """

        with csv_path.open(
            "r",
            encoding=
                "utf-8-sig",
            newline=
                "",
        ) as handle:

            reader = (
                csv.DictReader(
                    handle
                )
            )

            if (
                reader.fieldnames
                is None
            ):

                return (
                    []
                )

            rows = [
                {
                    str(
                        key
                    ):
                        (
                            ""
                            if value
                            is None
                            else str(
                                value
                            )
                        )

                    for (
                        key,
                        value,
                    )
                    in row.items()
                }

                for row
                in reader
            ]

        return (
            rows
        )

    # ==================================================================
    # COLUMN LOOKUP
    # ==================================================================

    @staticmethod
    def _column_name(
        row: dict[
            str,
            str,
        ],
        *candidates: str,
    ) -> str | None:
        """
        Resolve one prediction column case-insensitively.

        Current BirdNET acoustic prediction output includes:

            species_name
            confidence

        Additional aliases provide defensive compatibility.
        """

        normalized = {
            key
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            ):
                key

            for key
            in row
        }

        for candidate in (
            candidates
        ):

            normalized_candidate = (
                candidate
                .strip()
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            if (
                normalized_candidate
                in normalized
            ):

                return (
                    normalized[
                        normalized_candidate
                    ]
                )

        return (
            None
        )

    # ==================================================================
    # AGGREGATE SPECIES
    # ==================================================================

    def _aggregate_species_predictions(
        self,
        rows: Iterable[
            dict[
                str,
                str,
            ]
        ],
    ) -> tuple[
        BirdNETSpeciesPrediction,
        ...,
    ]:
        """
        Aggregate BirdNET detections by species.

        Aggregation
        -----------
        The maximum confidence observed for each species is retained.

        This avoids diluting a short call by averaging it with unrelated
        three-second windows where that call is absent.
        """

        maximum_scores: dict[
            str,
            float,
        ] = {}

        detection_counts: dict[
            str,
            int,
        ] = {}

        for row in (
            rows
        ):

            if not (
                row
            ):

                continue

            # ==========================================================
            # REQUIRED COLUMNS
            # ==========================================================

            species_column = (
                self._column_name(
                    row,
                    "species_name",
                    "species",
                    "label",
                )
            )

            confidence_column = (
                self._column_name(
                    row,
                    "confidence",
                    "score",
                )
            )

            if (
                species_column
                is None
                or confidence_column
                is None
            ):

                raise RuntimeError(
                    (
                        "BirdNET prediction table "
                        "does not contain expected "
                        "species/confidence columns."
                    )
                )

            species_name = (
                row[
                    species_column
                ]
                .strip()
            )

            if not (
                species_name
            ):

                continue

            try:

                confidence = float(
                    row[
                        confidence_column
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if (
                not math.isfinite(
                    confidence
                )
                or confidence
                < 0.0
            ):

                continue

            confidence = (
                min(
                    1.0,
                    confidence,
                )
            )

            # ==========================================================
            # DEFENSIVE THRESHOLD
            # ==========================================================
            #
            # BirdNET already receives this threshold directly.
            #
            # Keeping this test protects the project contract if a
            # future BirdNET release returns lower-confidence rows for
            # diagnostic reasons.
            # ==============================================================

            if (
                confidence
                < self.min_confidence
            ):

                continue

            # ==========================================================
            # MAXIMUM PER SPECIES
            # ==========================================================

            previous = (
                maximum_scores.get(
                    species_name
                )
            )

            if (
                previous
                is None
                or confidence
                > previous
            ):

                maximum_scores[
                    species_name
                ] = (
                    confidence
                )

            detection_counts[
                species_name
            ] = (
                detection_counts.get(
                    species_name,
                    0,
                )
                + 1
            )

        # ==============================================================
        # GLOBAL EVENT RANKING
        # ==============================================================

        ranking = sorted(
            maximum_scores.items(),

            key=
                lambda item: (
                    -item[
                        1
                    ],
                    item[
                        0
                    ],
                ),
        )

        ranking = (
            ranking[
                :
                self.top_k
            ]
        )

        return tuple(
            BirdNETSpeciesPrediction(
                species_name=
                    species_name,

                confidence=
                    confidence,

                detection_count=
                    detection_counts[
                        species_name
                    ],
            )

            for (
                species_name,
                confidence,
            )
            in ranking
        )

    # ==================================================================
    # SPECIES INFERENCE
    # ==================================================================

    def predict_species(
        self,
        classification_input: ClassificationInput,
    ) -> tuple[
        BirdNETSpeciesPrediction,
        ...,
    ]:
        """
        Run BirdNET and return ranked event-level species predictions.
        """

        self.validate_input(
            classification_input
        )

        if (
            classification_input.model_audio
            is None
        ):

            raise ValueError(
                (
                    "BirdNET requires "
                    "classification_input.model_audio."
                )
            )

        pcm16 = (
            self._audio_to_pcm16(
                classification_input
                .model_audio
            )
        )

        # ==============================================================
        # TEMPORARY INFERENCE DIRECTORY
        # ==============================================================

        with tempfile.TemporaryDirectory(
            prefix=
                "wildlife_birdnet_"
        ) as temporary_directory:

            root = (
                Path(
                    temporary_directory
                )
            )

            audio_path = (
                root
                / "event.wav"
            )

            prediction_path = (
                root
                / "predictions.csv"
            )

            # ==========================================================
            # WAV
            # ==============================================================

            self._write_wave_file(
                audio_path,
                pcm16,

                sample_rate=
                    classification_input
                    .sample_rate,
            )

            # ==========================================================
            # BIRDNET
            # ==============================================================

            prediction_result = (
                self._predict_file(
                    audio_path
                )
            )

            # ==========================================================
            # TABULAR INTERFACE
            # ==============================================================

            self._prediction_to_csv(
                prediction_result,
                prediction_path,
            )

            rows = (
                self._read_prediction_rows(
                    prediction_path
                )
            )

        return (
            self._aggregate_species_predictions(
                rows
            )
        )

    # ==================================================================
    # UNKNOWN / ABSTENTION RESULT
    # ==================================================================

    def _unknown_result(
        self,
        *,
        reasons: tuple[
            str,
            ...,
        ],
    ) -> ClassificationResult:
        """
        Build conservative UNKNOWN result.

        BirdNET no-detection is treated as abstention.

        It is not interpreted as 100% evidence for UNKNOWN because
        BirdNET is a specialist bird-species model rather than a complete
        classifier over all project acoustic categories.
        """

        return (
            ClassificationResult(
                label=
                    AcousticClass.UNKNOWN,

                confidence=
                    0.0,

                second_label=
                    None,

                second_confidence=
                    None,

                margin=
                    0.0,

                scores={
                    AcousticClass.BIRD.value:
                        0.0,

                    AcousticClass.UNKNOWN.value:
                        0.0,
                },

                reasons=
                    reasons,

                classifier_name=
                    self.name,

                classifier_version=
                    self.version,
            )
        )

    # ==================================================================
    # CLASSIFY
    # ==================================================================

    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:
        """
        Perform BirdNET species inference and adapt it to the project's
        broad classification contract.

        Accepted species prediction
        ---------------------------
        label:

            BIRD

        confidence:

            strongest accepted BirdNET species confidence

        second broad label:

            UNKNOWN

        broad margin:

            BIRD confidence - UNKNOWN score


        Species ambiguity
        -----------------
        Top-vs-second species margin is stored separately in:

            scores["birdnet:species_margin"]

        and explanatory reasons.
        """

        # ==============================================================
        # INPUT CONTRACT
        # ==============================================================

        self.validate_input(
            classification_input
        )

        # ==============================================================
        # SPECIES INFERENCE
        # ==============================================================

        predictions = (
            self.predict_species(
                classification_input
            )
        )

        # ==============================================================
        # NO ACCEPTED SPECIES
        # ==============================================================

        if not (
            predictions
        ):

            return (
                self._unknown_result(
                    reasons=(
                        (
                            "BirdNET produced no "
                            "species detection at or "
                            "above the configured "
                            "minimum confidence "
                            f"{self.min_confidence:.3f}"
                        ),

                        (
                            "no BirdNET detection is "
                            "treated as classifier "
                            "abstention rather than "
                            "strong evidence for another "
                            "broad acoustic class"
                        ),
                    )
                )
            )

        # ==============================================================
        # TOP SPECIES
        # ==============================================================

        top_prediction = (
            predictions[
                0
            ]
        )

        # ==============================================================
        # SECOND SPECIES / SPECIES MARGIN
        # ==============================================================

        second_species: (
            BirdNETSpeciesPrediction
            | None
        )

        if (
            len(
                predictions
            )
            >= 2
        ):

            second_species = (
                predictions[
                    1
                ]
            )

            species_margin = (
                top_prediction.confidence
                - second_species.confidence
            )

        else:

            second_species = (
                None
            )

            species_margin = (
                top_prediction.confidence
            )

        species_margin = float(
            max(
                0.0,
                min(
                    1.0,
                    species_margin,
                ),
            )
        )

        # ==============================================================
        # TAXONOMY RESOLUTION & BROAD SCORE MAP
        # ==============================================================

        broad_class = self.taxonomy.get_broad_class(
            top_prediction.species_name
        )

        taxon_group = self.taxonomy.get_taxon_group(
            top_prediction.species_name
        )

        if broad_class == AcousticClass.UNKNOWN:
            broad_label = AcousticClass.UNKNOWN
            broad_confidence = 0.0
            second_label = None
            second_confidence = None
            broad_margin = 0.0
        else:
            broad_label = broad_class
            broad_confidence = float(top_prediction.confidence)
            second_label = None
            second_confidence = None
            broad_margin = float(top_prediction.confidence)

        scores: dict[
            str,
            float,
        ] = {
            broad_label.value:
                broad_confidence,

            "birdnet:species_margin":
                species_margin,
        }

        if broad_label != AcousticClass.UNKNOWN:
            scores[AcousticClass.UNKNOWN.value] = 0.0

        # ==============================================================
        # SPECIES & ACOUSTIC SCORES
        # ==============================================================

        for prediction in (
            predictions
        ):

            scores[
                (
                    "species:"
                    f"{prediction.species_name}"
                )
            ] = (
                prediction.confidence
            )

            scores[
                (
                    "birdnet:acoustic:"
                    f"{prediction.species_name}"
                )
            ] = (
                prediction.confidence
            )

        # ==============================================================
        # GEOGRAPHIC PRIOR CONTEXT
        # ==============================================================

        geo_context_reason: str | None = None
        if self.geo_context.enabled:
            geo_model = self._ensure_geo_model()
            geo_priors = self.geo_context.query_geo_prior(geo_model)
            if geo_priors:
                for sp, prior_conf in geo_priors.items():
                    scores[f"birdnet:geo:{sp}"] = float(prior_conf)
                geo_context_reason = (
                    "BirdNET geographic priors were attached as separate "
                    "context scores and did not alter acoustic confidence"
                )
            elif self._geo_model_error is not None:
                geo_context_reason = (
                    "BirdNET geographic prior unavailable; acoustic "
                    f"classification continued: {self._geo_model_error}"
                )
            else:
                geo_context_reason = (
                    "BirdNET geographic context returned no prior scores "
                    "at or above its configured threshold"
                )

        # ==============================================================
        # EXPLANATION
        # ==============================================================

        reasons: list[
            str
        ] = [
            (
                "BirdNET top species prediction: "
                f"{top_prediction.species_name} "
                "(confidence="
                f"{top_prediction.confidence:.3f})"
            ),
        ]

        if geo_context_reason is not None:
            reasons.append(geo_context_reason)

        if broad_label != AcousticClass.UNKNOWN:
            reasons.append(
                (
                    f"structured taxonomy group '{taxon_group}' "
                    f"is mapped to broad class '{broad_label.value}'"
                )
            )
        else:
            reasons.append(
                (
                    "no verified broad taxonomy mapping is established "
                    "for this prediction -> safe abstention as UNKNOWN"
                )
            )

        if (
            top_prediction.common_name
            is not None
        ):

            reasons.append(
                (
                    "top common name: "
                    f"{top_prediction.common_name}"
                )
            )

        if (
            second_species
            is not None
        ):

            reasons.append(
                (
                    "second BirdNET species prediction: "
                    f"{second_species.species_name} "
                    "(confidence="
                    f"{second_species.confidence:.3f})"
                )
            )

            reasons.append(
                (
                    "top-vs-second species confidence "
                    f"margin={species_margin:.3f}"
                )
            )

        else:

            reasons.append(
                (
                    "no second accepted BirdNET "
                    "species prediction was available"
                )
            )

        reasons.append(
            (
                "ClassificationResult.margin="
                f"{broad_margin:.3f} represents the "
                f"broad {broad_label.name}-vs-UNKNOWN decision; "
                "species-level ambiguity is retained "
                "separately"
            )
        )

        reasons.append(
            (
                "species prediction is automated "
                "and requires independent validation "
                "before being treated as a verified "
                "wildlife observation"
            )
        )

        # ==============================================================
        # PROJECT CLASSIFICATION RESULT
        # ==============================================================

        return (
            ClassificationResult(
                label=
                    broad_label,

                confidence=
                    broad_confidence,

                second_label=
                    second_label,

                second_confidence=
                    second_confidence,

                margin=
                    broad_margin,

                scores=
                    scores,

                reasons=
                    tuple(
                        reasons
                    ),

                classifier_name=
                    self.name,

                classifier_version=
                    self.version,
            )
        )
