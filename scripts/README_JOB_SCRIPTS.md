# Scripts

Shell and Slurm launchers live in this directory.

Most launchers assume the project root is the current working directory and call
root Python entrypoints such as `python run_batch.py`. Use this pattern:

```bash
cd /path/to/skip_inv/src
bash scripts/run_batch.sh
```

Set output paths explicitly to the organized artifact tree when running new
experiments, for example `OUTPUT=../artifacts/outputs/my_run`.
