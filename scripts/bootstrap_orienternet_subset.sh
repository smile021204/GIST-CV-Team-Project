#!/usr/bin/env bash
set -euo pipefail

UPSTREAM="${1:-../OrienterNet}"

if [ ! -d "$UPSTREAM" ]; then
  echo "ERROR: upstream OrienterNet directory not found: $UPSTREAM"
  echo "Usage: bash scripts/bootstrap_orienternet_subset.sh /path/to/OrienterNet"
  exit 1
fi

echo "[1/4] Copying root files..."
cp -f "$UPSTREAM/setup.py" ./setup_upstream.py 2>/dev/null || true

echo "[2/4] Copying core maploc files..."
mkdir -p maploc
cp -f "$UPSTREAM/maploc/demo.py" maploc/demo.py
cp -f "$UPSTREAM/maploc/train.py" maploc/train.py
cp -f "$UPSTREAM/maploc/module.py" maploc/module.py

echo "[3/4] Copying required folders..."
mkdir -p maploc/models maploc/osm maploc/utils maploc/conf maploc/data maploc/evaluation
rsync -a --delete "$UPSTREAM/maploc/models/" maploc/models/
rsync -a --delete "$UPSTREAM/maploc/osm/" maploc/osm/
rsync -a --delete "$UPSTREAM/maploc/utils/" maploc/utils/
rsync -a "$UPSTREAM/maploc/conf/" maploc/conf/

cp -f "$UPSTREAM/maploc/data/dataset.py" maploc/data/dataset.py
cp -f "$UPSTREAM/maploc/data/image.py" maploc/data/image.py
cp -f "$UPSTREAM/maploc/data/torch.py" maploc/data/torch.py
cp -f "$UPSTREAM/maploc/data/utils.py" maploc/data/utils.py
cp -f "$UPSTREAM/maploc/data/sequential.py" maploc/data/sequential.py 2>/dev/null || true
mkdir -p maploc/data/mapillary
rsync -a "$UPSTREAM/maploc/data/mapillary/" maploc/data/mapillary/

cp -f "$UPSTREAM/maploc/evaluation/run.py" maploc/evaluation/run.py
cp -f "$UPSTREAM/maploc/evaluation/utils.py" maploc/evaluation/utils.py
cp -f "$UPSTREAM/maploc/evaluation/viz.py" maploc/evaluation/viz.py
cp -f "$UPSTREAM/maploc/evaluation/mapillary.py" maploc/evaluation/mapillary.py

echo "[4/4] Restoring custom GIST ABC files if overwritten..."
test -f maploc/conf/data/gist_abc.yaml || echo "WARNING: maploc/conf/data/gist_abc.yaml missing"
test -f maploc/data/gist_abc/dataset.py || echo "WARNING: maploc/data/gist_abc/dataset.py missing"
test -f maploc/evaluation/gist_abc.py || echo "WARNING: maploc/evaluation/gist_abc.py missing"

echo "Done. Next:"
echo "  1) Check maploc/data/__init__.py or data registry and add gist_abc if needed."
echo "  2) Edit maploc/data/gist_abc/dataset.py TODOs to match upstream sample schema."
echo "  3) Run python scripts/sanity_check_metadata.py --metadata datasets/gist_abc/metadata.csv"
