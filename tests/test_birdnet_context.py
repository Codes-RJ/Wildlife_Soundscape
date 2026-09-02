"""
Tests for BirdNET taxonomy and geographic context subsystem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wildlife_soundscape.classification.base import ClassificationInput
from wildlife_soundscape.classification.birdnet_backend import (
    BirdNETClassifierBackend,
)
from wildlife_soundscape.classification.birdnet_context import (
    BirdNETGeoContext,
    BirdNETTaxonomy,
    TAXON_GROUP_AMPHIBIA,
    TAXON_GROUP_AVES,
    TAXON_GROUP_INSECTA,
    TAXON_GROUP_MAMMALIA,
    TAXON_GROUP_NOISE,
    TAXON_GROUP_UNKNOWN,
    TAXON_TO_ACOUSTIC_CLASS,
)
from wildlife_soundscape.classification.classifier import AcousticClass, ClassificationResult


# ======================================================================
# TAXONOMY TESTS
# ======================================================================


def test_taxonomy_group_mapping_constants() -> None:
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_AVES] == AcousticClass.BIRD
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_INSECTA] == AcousticClass.INSECT
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_AMPHIBIA] == AcousticClass.AMPHIBIAN
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_MAMMALIA] == AcousticClass.MAMMAL
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_NOISE] == AcousticClass.NOISE
    assert TAXON_TO_ACOUSTIC_CLASS[TAXON_GROUP_UNKNOWN] == AcousticClass.UNKNOWN


def test_birdnet_taxonomy_lookup_and_registration() -> None:
    taxonomy = BirdNETTaxonomy()
    assert len(taxonomy) == 0
    assert taxonomy.get_broad_class("Corvus splendens_House Crow") == AcousticClass.UNKNOWN
    assert taxonomy.get_taxon_group("Corvus splendens_House Crow") is None

    taxonomy.register_species("Corvus splendens_House Crow", TAXON_GROUP_AVES)
    taxonomy.register_species("Gryllus bimaculatus", TAXON_GROUP_INSECTA)
    taxonomy.register_species("Rana temporaria_Common Frog", TAXON_GROUP_AMPHIBIA)
    taxonomy.register_species("Canis lupus_Grey Wolf", TAXON_GROUP_MAMMALIA)
    taxonomy.register_species("Engine_Noise", TAXON_GROUP_NOISE)

    assert len(taxonomy) >= 5
    assert taxonomy.get_broad_class("Corvus splendens_House Crow") == AcousticClass.BIRD
    assert taxonomy.get_broad_class("Gryllus bimaculatus") == AcousticClass.INSECT
    assert taxonomy.get_broad_class("Rana temporaria_Common Frog") == AcousticClass.AMPHIBIAN
    assert taxonomy.get_broad_class("Canis lupus_Grey Wolf") == AcousticClass.MAMMAL
    assert taxonomy.get_broad_class("Engine_Noise") == AcousticClass.NOISE

    # Scientific name prefix fallback
    assert taxonomy.get_broad_class("Corvus splendens") == AcousticClass.BIRD
    assert taxonomy.get_taxon_group("Corvus splendens") == TAXON_GROUP_AVES


def test_birdnet_taxonomy_csv_loading(tmp_path: Path) -> None:
    csv_file = tmp_path / "taxonomy.csv"
    csv_file.write_text(
        "species,taxon_group\n"
        "Turdus merula_Eurasian Blackbird,Aves\n"
        "Cicada orni,Insecta\n"
        "Bufo bufo,Amphibia\n",
        encoding="utf-8",
    )

    taxonomy = BirdNETTaxonomy.from_csv(csv_file)
    assert len(taxonomy) >= 3
    assert taxonomy.get_broad_class("Turdus merula_Eurasian Blackbird") == AcousticClass.BIRD
    assert taxonomy.get_broad_class("Cicada orni") == AcousticClass.INSECT
    assert taxonomy.get_broad_class("Bufo bufo") == AcousticClass.AMPHIBIAN


def test_birdnet_taxonomy_invalid_inputs() -> None:
    taxonomy = BirdNETTaxonomy()
    with pytest.raises(ValueError):
        taxonomy.register_species("", "Aves")
    with pytest.raises(ValueError):
        taxonomy.register_species("Corvus corax", "")


# ======================================================================
# GEOCONTEXT TESTS
# ======================================================================


def test_birdnet_geo_context_validation() -> None:
    # Disabled by default, allows None fields
    geo = BirdNETGeoContext()
    assert not geo.enabled

    # Enabled with valid params
    valid_geo = BirdNETGeoContext(
        enabled=True,
        latitude=12.9716,
        longitude=77.5946,
        week=24,
        min_confidence=0.05,
    )
    assert valid_geo.enabled
    assert valid_geo.latitude == 12.9716
    assert valid_geo.longitude == 77.5946
    assert valid_geo.week == 24

    # Invalid latitude
    with pytest.raises(ValueError):
        BirdNETGeoContext(enabled=True, latitude=100.0, longitude=0.0)

    # Invalid longitude
    with pytest.raises(ValueError):
        BirdNETGeoContext(enabled=True, latitude=0.0, longitude=200.0)

    # Invalid week
    with pytest.raises(ValueError):
        BirdNETGeoContext(enabled=True, latitude=0.0, longitude=0.0, week=50)

    # Invalid confidence
    with pytest.raises(ValueError):
        BirdNETGeoContext(enabled=True, latitude=0.0, longitude=0.0, min_confidence=-0.1)


def test_birdnet_geo_context_query_mock_model() -> None:
    geo = BirdNETGeoContext(
        enabled=True,
        latitude=12.97,
        longitude=77.59,
        week=10,
        min_confidence=0.1,
    )

    class FakeGeoModel:
        def predict(
            self,
            latitude: float,
            longitude: float,
            *,
            week: int | None = None,
            min_confidence: float = 0.03,
        ) -> dict[str, float]:
            assert latitude == 12.97
            assert longitude == 77.59
            assert week == 10
            assert min_confidence == 0.1
            return {
                "Corvus splendens_House Crow": 0.85,
                "Psittacula krameri_Rose-ringed Parakeet": 0.40,
                "Rare Species": 0.02,  # below min_confidence
            }

    priors = geo.query_geo_prior(FakeGeoModel())
    assert priors is not None
    assert "Corvus splendens_House Crow" in priors
    assert priors["Corvus splendens_House Crow"] == 0.85
    assert "Rare Species" not in priors  # filtered by min_confidence


def test_birdnet_geo_context_reads_official_tabular_result() -> None:
    geo = BirdNETGeoContext(
        enabled=True,
        latitude=12.97,
        longitude=77.59,
        min_confidence=0.1,
    )

    class FakeGeoResult:
        def to_csv(self, path: str) -> None:
            Path(path).write_text(
                "species_name,confidence\n"
                "Corvus splendens_House Crow,0.85\n"
                "Rare Species,0.02\n",
                encoding="utf-8",
            )

    class FakeGeoModel:
        def predict(self, *args, **kwargs) -> FakeGeoResult:
            return FakeGeoResult()

    assert geo.query_geo_prior(FakeGeoModel()) == {
        "Corvus splendens_House Crow": 0.85,
    }


# ======================================================================
# BIRDNET BACKEND INTEGRATION & ABSTENTION TESTS
# ======================================================================


def test_birdnet_backend_taxonomy_mapping_and_abstention() -> None:
    taxonomy = BirdNETTaxonomy()
    taxonomy.register_species("Corvus splendens_House Crow", TAXON_GROUP_AVES)
    taxonomy.register_species("Gryllus bimaculatus", TAXON_GROUP_INSECTA)

    class FakeDataFrame:
        def __init__(self, predictions: list[tuple[str, float]]) -> None:
            self._predictions = predictions

        def to_csv(self, path: Path | str) -> None:
            p = Path(path)
            lines = ["species_name,confidence"]
            for name, conf in self._predictions:
                lines.append(f"{name},{conf}")
            p.write_text("\n".join(lines), encoding="utf-8")

    # Mock BirdNET model producing custom prediction outputs
    class FakeBirdNETModel:
        def __init__(self, predictions: list[tuple[str, float]]) -> None:
            self._predictions = predictions

        def predict(self, *args, **kwargs) -> FakeDataFrame:
            return FakeDataFrame(self._predictions)

    # Test 1: Avian prediction mapped to BIRD
    backend_bird = BirdNETClassifierBackend(
        taxonomy=taxonomy,
        model=FakeBirdNETModel([("Corvus splendens_House Crow", 0.92)]),
        min_confidence=0.2,
    )
    audio = np.zeros(48000, dtype=np.float32)
    res_bird = backend_bird.classify(ClassificationInput(features=None, sample_rate=48000, model_audio=audio))
    assert isinstance(res_bird, ClassificationResult)
    assert res_bird.label == AcousticClass.BIRD
    assert res_bird.confidence == 0.92
    assert res_bird.second_label is None
    assert res_bird.second_confidence is None
    assert res_bird.scores["bird"] == 0.92
    assert res_bird.scores["species:Corvus splendens_House Crow"] == 0.92
    assert res_bird.scores["birdnet:acoustic:Corvus splendens_House Crow"] == 0.92

    # Test 2: Insect prediction mapped to INSECT
    backend_insect = BirdNETClassifierBackend(
        taxonomy=taxonomy,
        model=FakeBirdNETModel([("Gryllus bimaculatus", 0.88)]),
        min_confidence=0.2,
    )
    res_insect = backend_insect.classify(ClassificationInput(features=None, sample_rate=48000, model_audio=audio))
    assert res_insect.label == AcousticClass.INSECT
    assert res_insect.confidence == 0.88
    assert res_insect.second_label is None
    assert res_insect.second_confidence is None

    # Test 3: Unmapped taxon safely abstains as UNKNOWN with confidence 0
    backend_unknown = BirdNETClassifierBackend(
        taxonomy=taxonomy,
        model=FakeBirdNETModel([("Uncataloged_Mystery_Critter", 0.95)]),
        min_confidence=0.2,
    )
    res_unknown = backend_unknown.classify(ClassificationInput(features=None, sample_rate=48000, model_audio=audio))
    assert res_unknown.label == AcousticClass.UNKNOWN
    assert res_unknown.confidence == 0.0
    assert res_unknown.second_label is None
    assert res_unknown.second_confidence is None
    assert res_unknown.scores["species:Uncataloged_Mystery_Critter"] == 0.95
    assert "safe abstention" in " ".join(res_unknown.reasons)


def test_birdnet_backend_lazily_loads_and_traces_geo_model(monkeypatch) -> None:
    taxonomy = BirdNETTaxonomy({"Corvus splendens_House Crow": TAXON_GROUP_AVES})

    class FakeDataFrame:
        def to_csv(self, path: Path | str) -> None:
            Path(path).write_text(
                "species_name,confidence\nCorvus splendens_House Crow,0.92\n",
                encoding="utf-8",
            )

    class FakeAcousticModel:
        def predict(self, *args, **kwargs) -> FakeDataFrame:
            return FakeDataFrame()

    class FakeGeoModel:
        def predict(self, *args, **kwargs) -> dict[str, float]:
            return {"Corvus splendens_House Crow": 0.77}

    load_calls: list[tuple] = []

    class FakeBirdNETModule:
        @staticmethod
        def load(*args, **kwargs) -> FakeGeoModel:
            load_calls.append((*args, kwargs))
            return FakeGeoModel()

    backend = BirdNETClassifierBackend(
        taxonomy=taxonomy,
        model=FakeAcousticModel(),
        geo_context=BirdNETGeoContext(
            enabled=True,
            latitude=12.97,
            longitude=77.59,
            week=10,
        ),
    )
    monkeypatch.setattr(backend, "_load_birdnet_module", lambda: FakeBirdNETModule())

    model_input = ClassificationInput(
        features=None,
        sample_rate=48000,
        model_audio=np.zeros(48000, dtype=np.float32),
    )
    result = backend.classify(model_input)
    backend.classify(model_input)

    assert load_calls == [("geo", "3.0", "onnx", {"precision": "fp32"})]
    assert result.scores["birdnet:geo:Corvus splendens_House Crow"] == 0.77
    assert "did not alter acoustic confidence" in " ".join(result.reasons)


def test_birdnet_backend_geo_load_failure_preserves_acoustic_result(monkeypatch) -> None:
    taxonomy = BirdNETTaxonomy({"Corvus splendens_House Crow": TAXON_GROUP_AVES})

    class FakeDataFrame:
        def to_csv(self, path: Path | str) -> None:
            Path(path).write_text(
                "species_name,confidence\nCorvus splendens_House Crow,0.92\n",
                encoding="utf-8",
            )

    class FakeAcousticModel:
        def predict(self, *args, **kwargs) -> FakeDataFrame:
            return FakeDataFrame()

    backend = BirdNETClassifierBackend(
        taxonomy=taxonomy,
        model=FakeAcousticModel(),
        geo_context=BirdNETGeoContext(
            enabled=True,
            latitude=12.97,
            longitude=77.59,
        ),
    )

    def fail_load():
        raise RuntimeError("geo dependency unavailable")

    monkeypatch.setattr(backend, "_load_birdnet_module", fail_load)
    result = backend.classify(
        ClassificationInput(
            features=None,
            sample_rate=48000,
            model_audio=np.zeros(48000, dtype=np.float32),
        )
    )

    assert result.label == AcousticClass.BIRD
    assert result.confidence == 0.92
    assert "geo dependency unavailable" in " ".join(result.reasons)
