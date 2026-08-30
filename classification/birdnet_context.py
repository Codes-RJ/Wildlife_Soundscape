"""
BirdNET taxonomy and geographic context adapter.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Provide explicit biological taxonomy resolution and geographic/seasonal
prior context for BirdNET species-level predictions.

BirdNET V3 contains over 11,000 biological and acoustic classes spanning:

    Aves (birds)
    Insecta (insects)
    Amphibia (amphibians)
    Mammalia (mammals)
    Anthropogenic and environmental noise

This module ensures that:

    1. BirdNET predictions are mapped to broad project classes via
       explicit structured taxon metadata, NEVER via ad-hoc string parsing
       of scientific or common names.
    2. Non-bird classes are never silently mapped to AcousticClass.BIRD.
    3. Missing or ambiguous taxonomy metadata results in explicit safe
       abstention (AcousticClass.UNKNOWN).
    4. Optional geographic and seasonal occurrence priors (GeoModel) are
       traceable independently from raw acoustic inference.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .classifier import AcousticClass


# ======================================================================
# TAXONOMY GROUPS & MAPPINGS
# ======================================================================

TAXON_GROUP_AVES = "Aves"
TAXON_GROUP_INSECTA = "Insecta"
TAXON_GROUP_AMPHIBIA = "Amphibia"
TAXON_GROUP_MAMMALIA = "Mammalia"
TAXON_GROUP_NOISE = "Noise"
TAXON_GROUP_UNKNOWN = "Unknown"

# Standard mapping from biological taxon group to broad AcousticClass
TAXON_TO_ACOUSTIC_CLASS: Mapping[str, AcousticClass] = {
    TAXON_GROUP_AVES: AcousticClass.BIRD,
    TAXON_GROUP_INSECTA: AcousticClass.INSECT,
    TAXON_GROUP_AMPHIBIA: AcousticClass.AMPHIBIAN,
    TAXON_GROUP_MAMMALIA: AcousticClass.MAMMAL,
    TAXON_GROUP_NOISE: AcousticClass.NOISE,
    TAXON_GROUP_UNKNOWN: AcousticClass.UNKNOWN,
    # Common alternate casing / synonyms from standard datasets
    "aves": AcousticClass.BIRD,
    "bird": AcousticClass.BIRD,
    "birds": AcousticClass.BIRD,
    "insecta": AcousticClass.INSECT,
    "insect": AcousticClass.INSECT,
    "insects": AcousticClass.INSECT,
    "amphibia": AcousticClass.AMPHIBIAN,
    "amphibian": AcousticClass.AMPHIBIAN,
    "amphibians": AcousticClass.AMPHIBIAN,
    "mammalia": AcousticClass.MAMMAL,
    "mammal": AcousticClass.MAMMAL,
    "mammals": AcousticClass.MAMMAL,
    "noise": AcousticClass.NOISE,
    "anthropogenic": AcousticClass.NOISE,
    "geophony": AcousticClass.NOISE,
    "environmental": AcousticClass.NOISE,
    "soundscape": AcousticClass.NOISE,
}


# ======================================================================
# TAXONOMY ADAPTER
# ======================================================================

class BirdNETTaxonomy:
    """
    Structured taxonomy registry mapping BirdNET species labels to their
    verified biological taxon group and project AcousticClass.
    """

    def __init__(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        default_class: AcousticClass = AcousticClass.UNKNOWN,
    ) -> None:
        """
        Initialize the taxonomy registry.

        Parameters
        ----------
        mapping:
            Dictionary mapping exact BirdNET species labels (scientific name
            or 'Scientific_Common') to biological taxon group names
            (e.g., 'Aves', 'Insecta', 'Amphibia', 'Mammalia', 'Noise').
        default_class:
            AcousticClass returned when a species is not present in the mapping.
            Defaults to AcousticClass.UNKNOWN for safe abstention.
        """
        self._mapping: dict[str, str] = {}
        self._default_class = default_class
        if mapping:
            for k, v in mapping.items():
                self.register_species(k, v)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        species_column: str = "species",
        taxon_column: str = "taxon_group",
    ) -> BirdNETTaxonomy:
        """
        Load taxonomy mapping from a structured CSV file.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Taxonomy CSV not found: {path}")

        instance = cls()
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = row.get(species_column, "").strip()
                taxon = row.get(taxon_column, "").strip()
                if species and taxon:
                    instance.register_species(species, taxon)
        return instance

    def register_species(self, species_label: str, taxon_group: str) -> None:
        """
        Register or update a species taxon group mapping.
        """
        if not species_label or not isinstance(species_label, str):
            raise ValueError("species_label must be a non-empty string")
        if not taxon_group or not isinstance(taxon_group, str):
            raise ValueError("taxon_group must be a non-empty string")
        label = species_label.strip()
        taxon = taxon_group.strip()
        self._mapping[label] = taxon
        if "_" in label:
            sci_name = label.split("_", 1)[0].strip()
            if sci_name:
                self._mapping[sci_name] = taxon

    def get_taxon_group(self, species_label: str) -> str | None:
        """
        Look up the biological taxon group for an exact species label.
        """
        if not species_label:
            return None
        clean_label = species_label.strip()
        if clean_label in self._mapping:
            return self._mapping[clean_label]
        # Also try matching just the scientific name part before '_'
        if "_" in clean_label:
            sci_name = clean_label.split("_", 1)[0].strip()
            if sci_name in self._mapping:
                return self._mapping[sci_name]
        return None

    def get_broad_class(self, species_label: str) -> AcousticClass:
        """
        Resolve the broad AcousticClass for a BirdNET prediction label.

        Returns AcousticClass.UNKNOWN if no mapping is established.
        """
        taxon_group = self.get_taxon_group(species_label)
        if taxon_group is None:
            return self._default_class
        return TAXON_TO_ACOUSTIC_CLASS.get(taxon_group, self._default_class)

    def is_known(self, species_label: str) -> bool:
        """
        Check if a species label has an established taxonomy mapping.
        """
        return self.get_taxon_group(species_label) is not None

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"BirdNETTaxonomy(entries={len(self._mapping)}, default={self._default_class.name})"


# ======================================================================
# GEOGRAPHIC / TEMPORAL CONTEXT
# ======================================================================

@dataclass(frozen=True, slots=True)
class BirdNETGeoContext:
    """
    Configuration and state for BirdNET geographic and seasonal prior modeling.
    """

    enabled: bool = False
    latitude: float | None = None
    longitude: float | None = None
    week: int | None = None  # 1 to 48 in BirdNET convention (4 weeks/month)
    min_confidence: float = 0.03

    def __post_init__(self) -> None:
        if self.enabled:
            if self.latitude is None or not math.isfinite(self.latitude):
                raise ValueError("GeoContext enabled requires finite latitude.")
            if not (-90.0 <= self.latitude <= 90.0):
                raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}.")

            if self.longitude is None or not math.isfinite(self.longitude):
                raise ValueError("GeoContext enabled requires finite longitude.")
            if not (-180.0 <= self.longitude <= 180.0):
                raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}.")

            if self.week is not None:
                if not (1 <= self.week <= 48):
                    raise ValueError(f"Week must be in [1, 48], got {self.week}.")

            if not math.isfinite(self.min_confidence) or self.min_confidence < 0.0:
                raise ValueError(f"min_confidence must be non-negative, got {self.min_confidence}.")

    def query_geo_prior(
        self,
        geo_model: Any | None = None,
    ) -> dict[str, float] | None:
        """
        Query BirdNET GeoModel if available and return occurrence probabilities.

        Returns None if disabled, geo_model is unavailable, or query fails.
        """
        if not self.enabled or geo_model is None:
            return None

        if self.latitude is None or self.longitude is None:
            return None

        try:
            kwargs: dict[str, Any] = {
                "latitude": self.latitude,
                "longitude": self.longitude,
            }
            if self.week is not None:
                kwargs["week"] = self.week

            # Query official birdnet GeoModel predict method
            if hasattr(geo_model, "predict"):
                predictions = geo_model.predict(**kwargs)
                if isinstance(predictions, dict):
                    return {
                        str(k): float(v)
                        for k, v in predictions.items()
                        if float(v) >= self.min_confidence
                    }
                elif isinstance(predictions, (list, tuple)):
                    res: dict[str, float] = {}
                    for item in predictions:
                        if hasattr(item, "species_name") and hasattr(item, "confidence"):
                            if item.confidence >= self.min_confidence:
                                res[str(item.species_name)] = float(item.confidence)
                    return res
        except Exception:
            return None

        return None
