#!/usr/bin/env bash
set -euo pipefail

# Usage: bash examples/refit_stage2.sh [N_SOURCES] (default: 1000).
n_sources="${1:-1000}"
if (( $# > 1 )) || ! [[ "$n_sources" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash examples/refit_stage2.sh [positive source count]" >&2
  exit 2
fi

# Run from the repository root with the three Zenodo archives in zenodo/.
tar -xzf zenodo/cosmology_closure_inputs.tar.gz -C zenodo
tar -xzf zenodo/cosmology_closure_stage2.tar.gz -C zenodo
tar -xzf zenodo/cosmology_closure_stage1_prior_5.tar.gz -C zenodo

# Sample the requested number from the paper's Prior 5 selection.
python - "$n_sources" <<'PY'
from pathlib import Path
import sys
import numpy as np

prior = 5
n_sources = int(sys.argv[1])
base = Path('zenodo/results/cosmology_closure')
run = base / f'prior_{prior}' / 'stage2'
with np.load(run / 'selected_source_ids.npz', allow_pickle=False) as archive:
    ids = archive['source_id_selected']
rng = np.random.default_rng(42)
if not 1 <= n_sources <= len(ids):
    raise ValueError(f"Choose between 1 and {len(ids):,} sources; requested {n_sources:,}.")
selected = np.sort(rng.choice(ids, size=n_sources, replace=False))
out = Path('reanalysis')
out.mkdir(exist_ok=True)
np.savez(out / f'prior_{prior}_subsample_source_ids.npz', source_id_selected=selected)
print(f'Selected {len(selected):,} of {len(ids):,} available Prior {prior} sources.')
PY

# Refit with the paper's parameters, fixed values and draw selection.
# To run the fit command in Apptainer, use --pwd /opt/agn and mount
# zenodo/ read-only at /opt/agn/zenodo and reanalysis/ at /opt/agn/reanalysis.
PYTHONPATH=code python -m framework.stage2_fitting \
  --chains-npz zenodo/results/cosmology_closure/prior_5/stage1/posterior_samples.npz \
  --chains-npz-original zenodo/results/cosmology_closure/prior_5/stage1/posterior_samples.npz \
  --source-id-manifest reanalysis/prior_5_subsample_source_ids.npz \
  --pair-npz zenodo/results/cosmology_closure/simulated_sources.npz \
  --band gaia_g --z-key redsh --logl3000-key logL3000 \
  --m-win-key posterior_m_win_cont_Gaia_G --lnD-key posterior_ln_D_Gaia_G \
  --sed-grid assets/sed/quasar_continuum_and_lines.npz \
  --passband-file assets/passbands/gaia_g.dat \
  --z-min 0.5 --z-max 3.5 --logl3000-min=-inf --logl3000-max=inf \
  --d50-min=-6 --d50-max=-2.5 \
  --draw-stride 2 --draw-offset 0 --nan-policy finite-paired-draws --min-finite-draws 250 \
  --sample-params Om0,w0,wa,D_pivot,d_D,log_sigma_int \
  --H0-fixed 67.4 --alpha-D-fixed 2.51 \
  --warmup 3000 --samples 1500 --chains 4 --chain-method sequential --seed 123 \
  --target-accept-prob 0.95 --max-tree-depth 11 --n-steps-z 1024 \
  --output reanalysis/prior_5_subsample_stage2.npz
