#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with observed_gaia_light_curves.tar.gz
# extracted into zenodo/.
# Generate 33,600 mock light curves from 800 Gaia templates and 42 DRW pairs.
# This draws new templates from the released cleaned pool. The exact paper
# curves are supplied in simulated_light_curves.tar.gz.
mkdir -p results/recovery_example
PYTHONPATH=code python -m framework.generate_recovery_grid \
  --input-fits zenodo/external_data/observed_gaia_light_curves.fits \
  --output-fits results/recovery_example/mock_light_curves.fits \
  --tau-days 5,20,80,300,1000,3000,10000 \
  --sigma-mag 0.03,0.06,0.10,0.14,0.20,0.30 \
  --m0-bin-edges 19.0,19.8,20.3 --npts-bin-edges 29,40,52 \
  --templates-per-bin 50 --min-epochs 10 --seed 42

# Fit Prior 5. To refit the paper's mocks, skip generation and pass
# zenodo/external_data/prior_sensitivity_light_curves.fits to --input-fits.
PYTHONPATH=code python -m framework.stage1_fitting \
  --input-fits results/recovery_example/mock_light_curves.fits \
  --output-npz results/recovery_example/prior_5_posterior.npz \
  --bands Gaia_G --sampling-model lnD_lntau_flat \
  --prior-lnD-min -35 --prior-lnD-max 26 \
  --prior-lntau-min -6 --prior-lntau-max 15 \
  --nwalkers 32 --nburn 500 --nsteps 400 --nsamples-output 2000 \
  --adaptive --target-thinned 2000 --step-chunk 500 --max-steps 24000 \
  --mwin-mode cont --seed 123
