# Spec — Dependency Policy

Core project should remain usable without optional research components.

## BirdNET
Keep optional/lazy loading behavior.

## GIS
Prefer optional:
- `pyproj>=3.7,<4`

If project policy avoids new core dependencies, put it in an optional
requirements file such as:
- `requirements-research.txt`

## Embedding clustering — later only
Optional future dependencies:
- umap-learn
- hdbscan

Do not add them until the optional embedding phase is implemented.
