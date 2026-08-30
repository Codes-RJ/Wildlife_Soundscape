# Spec — BirdNET Taxonomy + Geographic Context

## Mandatory correction

BirdNET V3 predictions must not be automatically mapped to broad `BIRD`.

Create a context/taxonomy adapter that can reliably determine broad group.

Preferred broad mappings:
- Aves -> AcousticClass.BIRD
- Insecta -> AcousticClass.INSECT
- Amphibia -> AcousticClass.AMPHIBIAN
- Mammalia -> AcousticClass.MAMMAL
- Anthropogenic/Geophony sound classes -> AcousticClass.NOISE
- unsupported/ambiguous -> AcousticClass.UNKNOWN

## Taxonomy source

Use a trusted BirdNET-compatible taxonomy metadata source. Prefer structured
metadata with a taxon-group field.

Do NOT:
- guess from common names;
- guess from scientific-name suffixes;
- assume every BirdNET class is avian.

## GeoModel

Optional configuration:
- enabled
- latitude
- longitude
- week/date
- min occurrence confidence

Keep acoustic score and geo occurrence score separately in traceability fields.

Suggested namespaced scores:
- `species:<label>`
- `birdnet:acoustic:<label>`
- `birdnet:geo:<label>`
- `birdnet:species_margin`

If a combined project-specific score is created, name it explicitly, e.g.:
- `birdnet:fused:<label>`

and document the formula.

## Abstention

When no broad mapping can be justified:
- label UNKNOWN;
- confidence 0 or conservative project-defined evidence;
- if `second_label=None`, `second_confidence=None`.

Do not manufacture UNKNOWN certainty.
