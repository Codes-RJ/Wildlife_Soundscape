import pytest

from wildlife_soundscape.classification.base import (
    ClassifierBackend,
    ClassificationInput,
)
from wildlife_soundscape.classification.classifier import (
    AcousticClass,
    ClassificationResult,
)
from wildlife_soundscape.classification.ensemble_backend import (
    EnsembleClassifierBackend,
    EnsembleMember,
)


class Backend(ClassifierBackend):
    def __init__(self, name, label=AcousticClass.BIRD, fail=False):
        self._name, self.label, self.fail = name, label, fail

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return "test-1"

    @property
    def requires_audio(self):
        return False

    @property
    def requires_features(self):
        return False

    def classify(self, classification_input):
        if self.fail:
            raise RuntimeError("model unavailable")
        return ClassificationResult(
            self.label,
            0.9,
            None,
            None,
            0.9,
            {self.label.value: 0.9},
            (),
            self.name,
            self.version,
        )


def test_ensemble_keeps_success_and_failure_provenance():
    backend = EnsembleClassifierBackend(
        [Backend("working"), Backend("missing", fail=True)]
    )
    result = backend.classify(ClassificationInput(None, 48000))
    assert result.label == AcousticClass.BIRD
    assert any("missing" in reason and "failed" in reason for reason in result.reasons)


def test_ensemble_weights_change_decision():
    backend = EnsembleClassifierBackend(
        [
            EnsembleMember(Backend("bird"), weight=1),
            EnsembleMember(Backend("insect", AcousticClass.INSECT), weight=3),
        ]
    )
    assert (
        backend.classify(ClassificationInput(None, 48000)).label == AcousticClass.INSECT
    )


@pytest.mark.parametrize("strict", [False, True])
def test_ensemble_never_invents_success_when_models_fail(strict):
    backend = EnsembleClassifierBackend(
        [Backend("one", fail=True), Backend("two", fail=True)],
        continue_on_member_error=not strict,
    )
    with pytest.raises(RuntimeError):
        backend.classify(ClassificationInput(None, 48000))
