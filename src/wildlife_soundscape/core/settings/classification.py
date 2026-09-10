from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .validation import _require_finite, _require_positive_int


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationConfig:
    """
    Acoustic-classification subsystem configuration.

    Current operational backend:

        heuristic

    Reserved future backends:

        pretrained
        birdnet
        ensemble
    """

    enabled: bool = True

    backend: str = "heuristic"

    fallback_to_heuristic: bool = True

    provide_model_audio: bool = True

    model_path: Path | None = None

    labels_path: Path | None = None

    model_sample_rate: int | None = None

    top_k: int = 5

    model_min_confidence: float = 0.20

    latitude: float | None = None

    longitude: float | None = None

    week: int | None = None

    use_geo_filter: bool = False

    geo_min_confidence: float = 0.03

    taxonomy_path: Path | None = None

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate classification configuration.

        Model files are intentionally not required to exist while the
        heuristic backend is active.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):
            raise TypeError(("Classification enabled must be bool."))

        if not isinstance(
            self.backend,
            str,
        ):
            raise TypeError(("Classification backend must be a string."))

        backend = self.backend.strip().lower()

        if not (backend):
            raise ValueError(("Classification backend cannot be empty."))

        supported_backends = {
            "heuristic",
            "pretrained",
            "birdnet",
            "ensemble",
        }

        if backend not in supported_backends:
            raise ValueError(
                (
                    "Unsupported classification "
                    f"backend '{self.backend}'. "
                    "Supported values: "
                    f"{sorted(supported_backends)}"
                )
            )

        if not isinstance(
            self.fallback_to_heuristic,
            bool,
        ):
            raise TypeError(("Classification fallback_to_heuristic must be bool."))

        if not isinstance(
            self.provide_model_audio,
            bool,
        ):
            raise TypeError(("Classification provide_model_audio must be bool."))

        if self.model_path is not None and not isinstance(
            self.model_path,
            Path,
        ):
            raise TypeError(("Classification model_path must be Path or None."))

        if self.labels_path is not None and not isinstance(
            self.labels_path,
            Path,
        ):
            raise TypeError(("Classification labels_path must be Path or None."))

        if self.model_sample_rate is not None:
            _require_positive_int(
                self.model_sample_rate,
                name=("Classification model_sample_rate"),
            )

        _require_positive_int(
            self.top_k,
            name="Classification top_k",
        )

        minimum_confidence = _require_finite(
            self.model_min_confidence,
            name=("Classification model_min_confidence"),
        )

        if not (0.0 <= minimum_confidence <= 1.0):
            raise ValueError(
                ("Classification model_min_confidence must be between 0 and 1.")
            )

        if not isinstance(
            self.use_geo_filter,
            bool,
        ):
            raise TypeError(("Classification use_geo_filter must be bool."))

        if self.latitude is not None:
            lat = _require_finite(
                self.latitude,
                name="Classification latitude",
            )
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Classification latitude must be in [-90, 90], got {lat}."
                )

        if self.longitude is not None:
            lon = _require_finite(
                self.longitude,
                name="Classification longitude",
            )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Classification longitude must be in [-180, 180], got {lon}."
                )

        if self.week is not None:
            _require_positive_int(
                self.week,
                name="Classification week",
            )
            if not (1 <= self.week <= 48):
                raise ValueError(
                    f"Classification week must be between 1 and 48, got {self.week}."
                )

        geo_min_conf = _require_finite(
            self.geo_min_confidence,
            name="Classification geo_min_confidence",
        )
        if not (0.0 <= geo_min_conf <= 1.0):
            raise ValueError(
                f"Classification geo_min_confidence must be in [0, 1], got {geo_min_conf}."
            )

        if self.taxonomy_path is not None and not isinstance(self.taxonomy_path, Path):
            raise TypeError("Classification taxonomy_path must be Path or None.")
