# Source Directory

This directory is intended to be the uploadable code subset.

It contains root Python entrypoints plus `utils/`, `experiments/`, `analysis/`,
`p2p/`, `scripts/`, and related code documentation. Run commands from this
directory unless you explicitly pass absolute paths.

`PIE_bench` is a relative symlink to `../misc/PIE_bench` so old defaults such as
`PIE_bench/mapping_file.json` continue to work locally.
