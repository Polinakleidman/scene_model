#!/bin/bash
# Final step of the SIMBA/TCR run (submitted as a ws-ia job from reproduction/):
#  1. SCENE-TCR module evaluation with 12 workers. permutation_rng=independent gives every module
#     its own seed stream, so results do not depend on n_jobs (the queued copy would take ~8 h on 1 core).
#  2. Wait for job B to finish the SIMBA-TCR evaluation, then cancel B (its last command is step 1).
#  3. Restore SIMBA in the fig_4/fig_6 notebooks and render all figures and tables.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
B_JOB=${B_JOB:?set B_JOB}
cd SCENE-reproducibility
echo "$(date +%T) SCENE-TCR evaluation (12 workers)"
../.venv/bin/python scripts/eval_gene_module.py --input_h5ad data/tcr_stim_data.h5ad --embedding_npy results/geometric_biology/scLDM_TCR/scLDM_gene_latent.npy --embedding_gene_names_npy results/geometric_biology/scLDM_TCR/scLDM_gene_names.npy --module_json data/tf_regulons.json --output_csv results/geometric_biology/scLDM_TCR/gene_module_eval.csv --n_perm 10000 --seed 42 --n_jobs 12 --permutation_rng independent
cd ..
until grep -qE "\[simba-B\] (OK|FAIL) +57/58" logs/queue.log || ! squeue -j "$B_JOB" -h | grep -q .; do sleep 60; done
grep -E "\[simba-B\] (OK|FAIL) +57/58" logs/queue.log
scancel "$B_JOB" 2>/dev/null; echo "$(date +%T) cancelled B ($B_JOB)"
cd SCENE-reproducibility
git checkout -- notebooks/figures/fig_4_benchmarks.ipynb notebooks/figures/fig_6_plot_modules.ipynb
N=notebooks/figures
../.venv/bin/python ../render_allow_errors.py $N/fig_*.ipynb $N/app_low_dim_comparison.ipynb $N/app_imputation.ipynb
../.venv/bin/python scripts/write_paper_tables.py
echo "$(date +%T) DONE"
