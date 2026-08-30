# Spec — GeoJSON Export

## Required configuration

- origin latitude
- origin longitude
- array azimuth degrees from true north

## Coordinate convention

First document the existing local x/y convention.

Then convert:
1. local array coordinates
2. rotate by array azimuth
3. East/North offsets
4. projected/geodetic conversion
5. WGS84 lon/lat

Prefer `pyproj` for correctness.

## GeoJSON

Output:
- FeatureCollection
- one Point Feature per successfully localized event

Properties should remain ordinary JSON types.

Do not export fabricated GPS points for unlocalized events.

## Privacy/research note

If future fieldwork involves sensitive/endangered species, add an option to
coarsen or omit exact coordinates from public exports.
