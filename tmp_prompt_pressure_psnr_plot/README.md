# Sensitive Prompt Pressure vs PSNR

This directory contains the formal analysis outputs for the sensitive prompt pressure experiment.

- Rows: `100` seeds
- FPI manifest: `../artifacts/outputs/prompt_group_top1_fpi_100/sensitive/manifest.csv`
- DDIM pressure manifest: `../artifacts/outputs/sensitive_prompt_pressure_ddim_inv_rec/manifest.csv`
- X axis: `prompt_pressure_total` from the DDIM prompt-pressure generation run.
- Y axis: reconstruction PSNR.
- FPI Pearson r: `-0.289512`
- DDIM Pearson r: `-0.312362`
