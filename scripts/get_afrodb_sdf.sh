#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/external
# Option A: Figshare (AfroDb, PLOS One). If blocked, use COCONUT below.
echo "Attempting AfroDb Figshare..."
echo "Open this in a browser if curl fails: (see paper’s Figshare)."
# Placeholder; Figshare often changes signed URLs. Use browser then:
# mv ~/Downloads/AfroDb*.sdf data/external/afrodb_3d_structures.sdf

# Option B: COCONUT (download an SDF of a query or the full set)
echo "COCONUT alternative: visit the site and export SDF for your query."
echo "https://coconut.naturalproducts.net  (export -> SDF)"
