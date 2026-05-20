#!/usr/bin/env python3
"""Create a lightweight Leaflet HTML map from metadata CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    points = []
    with args.metadata.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except Exception:
                continue
            points.append({
                "lat": lat,
                "lon": lon,
                "building_id": row.get("building_id", ""),
                "image_path": row.get("image_path", ""),
                "yaw": row.get("yaw", ""),
            })

    if not points:
        raise SystemExit("No valid GPS points found.")

    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>GIST ABC GPS points</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #map {{ width: 100vw; height: 100vh; }}
  .legend {{ position: absolute; z-index: 999; top: 10px; right: 10px; background: white; padding: 8px; border-radius: 6px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">GIST ABC GPS points<br/>A/B/C from folder or metadata</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const points = {json.dumps(points, ensure_ascii=False)};
const map = L.map('map').setView([{center_lat}, {center_lon}], 18);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 20,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

const colors = {{A: 'red', B: 'blue', C: 'green'}};

points.forEach(p => {{
  const color = colors[p.building_id] || 'black';
  const marker = L.circleMarker([p.lat, p.lon], {{
    radius: 5,
    color,
    fillColor: color,
    fillOpacity: 0.75,
  }}).addTo(map);
  marker.bindPopup(`<b>${{p.building_id}}</b><br>${{p.image_path}}<br>yaw=${{p.yaw}}`);
}});
</script>
</body>
</html>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(f"Wrote {args.out} with {len(points)} points")


if __name__ == "__main__":
    main()
