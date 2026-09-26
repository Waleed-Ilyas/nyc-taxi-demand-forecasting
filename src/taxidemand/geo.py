"""Convert the official TLC taxi-zone shapefile to a small GeoJSON for the app map.

Usage:  python -m taxidemand.geo     (needs the dev extras: pyshp, pyproj, shapely)
"""
from __future__ import annotations

import json
import zipfile

from . import config as C


def build_geojson(tolerance_deg: float = 0.0003) -> dict:
    import shapefile  # pyshp
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform

    with zipfile.ZipFile(C.RAW_DIR / "taxi_zones.zip") as z:
        z.extractall(C.RAW_DIR / "taxi_zones")
    shp = next((C.RAW_DIR / "taxi_zones").rglob("*.shp"))
    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    to_wgs = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True).transform
    features = []
    for sr in reader.shapeRecords():
        rec = dict(zip(fields, sr.record, strict=True))
        geom = transform(to_wgs, shape(sr.shape.__geo_interface__)).simplify(
            tolerance_deg, preserve_topology=True)
        features.append({"type": "Feature", "id": int(rec["LocationID"]),
                         "properties": {"LocationID": int(rec["LocationID"]), "zone": rec["zone"],
                                        "borough": rec["borough"]},
                         "geometry": json.loads(json.dumps(mapping(geom)))})
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    gj = build_geojson()
    # round coordinates to 5 decimals (~1 m) to keep the file small
    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 5), round(c[1], 5)]
    for f in gj["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    out = C.ARTIFACTS_DIR / "zones.geojson"
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(gj, separators=(",", ":")))
    print(f"{len(gj['features'])} zones -> {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
