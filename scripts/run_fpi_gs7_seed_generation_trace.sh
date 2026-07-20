#!/bin/bash

# Backward-compatible entry point.
# FPI tests now load saved generated latents instead of regenerating source images.

PROJECT_DIR=${PROJECT_DIR:-$PWD}
exec bash scripts/run_fpi_gs7_seed_from_saved_latents.sh "$@"
