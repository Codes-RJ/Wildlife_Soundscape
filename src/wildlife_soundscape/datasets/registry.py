"""Curated datasets and their permitted role in this project."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecommendedDataset:
    """One dataset recommendation with explicit compatibility limits."""

    dataset_id: str
    name: str
    url: str | None
    tasks: tuple[str, ...]
    region: str
    synchronized_array: bool
    final_system_validation: bool
    role: str


RECOMMENDED_DATASETS = (
    RecommendedDataset(
        dataset_id="native-wildlife-array-v1",
        name="Native Wildlife Soundscape Array Dataset",
        url=None,
        tasks=(
            "event_detection",
            "classification",
            "localization",
            "ecoacoustics",
        ),
        region="deployment site",
        synchronized_array=True,
        final_system_validation=True,
        role="Required final dataset recorded by the deployed three-node array.",
    ),
    RecommendedDataset(
        dataset_id="birdclef-2024",
        name="BirdCLEF 2024",
        url="https://www.kaggle.com/competitions/birdclef-2024",
        tasks=("event_detection", "classification"),
        region="Western Ghats, India",
        synchronized_array=False,
        final_system_validation=False,
        role="Primary public bird benchmark for Western Ghats deployments.",
    ),
    RecommendedDataset(
        dataset_id="beans",
        name="BEANS",
        url="https://github.com/earthspecies/beans",
        tasks=("event_detection", "classification"),
        region="multiple",
        synchronized_array=False,
        final_system_validation=False,
        role="Broad-taxon regression benchmark; select relevant subsets.",
    ),
    RecommendedDataset(
        dataset_id="locata",
        name="LOCATA",
        url="https://www.locata.lms.tf.fau.de/datasets/",
        tasks=("localization",),
        region="indoor speech",
        synchronized_array=True,
        final_system_validation=False,
        role="External localization sanity check with different arrays and sources.",
    ),
    RecommendedDataset(
        dataset_id="starss23",
        name="STARSS23",
        url=(
            "https://dcase.community/challenge2024/"
            "task-audio-and-audiovisual-sound-event-localization-and-"
            "detection-with-source-distance-estimation"
        ),
        tasks=("event_detection", "localization"),
        region="indoor spatial scenes",
        synchronized_array=True,
        final_system_validation=False,
        role="Four-channel spatial-event comparison, not native-array validation.",
    ),
    RecommendedDataset(
        dataset_id="a2o",
        name="Australian Acoustic Observatory",
        url="https://acousticobservatory.org/data/",
        tasks=("ecoacoustics", "event_detection"),
        region="Australia",
        synchronized_array=False,
        final_system_validation=False,
        role="Real continuous-audio workflow benchmark with geographic mismatch.",
    ),
)
