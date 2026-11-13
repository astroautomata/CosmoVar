#!/usr/bin/env python3
"""Reproduce the Section 5.1 PSD robustness tests.

- Inputs: observed Gaia light curves and Prior 1/5 Stage 1 fits from
  reproduce_from_light_curves.py.
- Default: print the commands; add --execute to run them.
- Tests: fixed slope/bend grid, luminosity-dependent cases, including
  the stronger s=4 cases, and the Zw 229–15 benchmark.
- Outputs: fitted results, plot_data/ tables and psd_steepening.pdf
  under --output.

To replot without fitting, place psd_steepening_figure.tar.gz in zenodo/
and run from the repository root:

mkdir -p zenodo/results/figure_analysis/psd_steepening
tar -xzf zenodo/psd_steepening_figure.tar.gz \
  --strip-components=3 -C zenodo/results/figure_analysis/psd_steepening \
  --wildcards 'figures/psd_steepening/data/*'
PYTHONPATH=code python -m plotting.plot_psd_steepening \
  --output figures/psd_steepening_rebuilt.pdf

Default samples: 8,200 representative templates and up to 2,500
common-support templates for the fixed grid and Zw benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "representative": ("representative", "drw,popR,popLup,popLdown,popSLdown,popSLup"),
    "fixed_grid": ("common_support", "drw," + ",".join(
        f"s{s}_tb{t}" for s in ("3", "3.4", "4")
        for t in ("0.08333", "0.25", "0.5", "1", "2", "3"))),
    "stronger_luminosity": ("representative", "drw,popLup_s4,popLdown_s4"),
    "zw229_15": ("common_support", "drw,s4.51_tb0.884"),
}


def pack_results(paths, data):
    """Save reduced tables for the PSD figure renderer.

    Rows pair PSD variants and DRW controls by template ID (tid).
    eta16/eta50/eta84 are log10(D) posterior quantiles in dex.
    tpl_z is redshift; tpl_tau is the injected rest-frame damping time
    in days; tpl_ell is catalogue log10(L3000), with L3000 in erg/s.
    summary.json stores paired shifts, regression coefficients and redshift bins.
    """
    import numpy as np
    data.mkdir(parents=True, exist_ok=False)
    summary = {"rows": [], "inputs": {}}
    fields = ("ok", "tid", "variant_idx", "eta16", "eta50", "eta84", "log10D_true")
    for name, folder in paths.items():
        config = json.loads((folder / "psd_summary.json").read_text())
        with np.load(folder / "psd_results.npz", allow_pickle=True) as raw:
            variants = raw["variants"].astype(str)
            keep = np.arange(len(variants))
            if name == "fixed_grid":
                keep = np.flatnonzero([v == "drw" or v.startswith("s") for v in variants])
            mask = np.isin(raw["variant_idx"], keep)
            arrays = {key: raw[key][mask] for key in fields}
            arrays["variant_idx"] = np.searchsorted(keep, arrays["variant_idx"])
            variants = variants[keep]
            arrays.update(variants=variants, tpl_z=raw["tpl_z"],
                          tpl_tau=raw["tpl_tau"], tpl_ell=raw["tpl_ell"])
            if name == "representative":
                lags = []
                for times, z in zip(raw["tpl_tobs"], arrays["tpl_z"]):
                    rest = (times - times[0]) / (1 + z)
                    delta = np.abs(rest[:, None] - rest[None, :])[np.triu_indices(len(rest), 1)]
                    lags.append(delta[delta > 0])
                edges = np.logspace(-2.3, 3, 54)
                arrays["pair_lag_counts"], arrays["pair_lag_edges_days"] = np.histogram(np.concatenate(lags), bins=edges)
                summary["z_edges"] = config["z_edges"]
            np.savez_compressed(data / f"{name}.npz", **arrays)
        for row in config["rows"]:
            variant = row["variant"]
            if variant == "drw" or variant not in variants or variant in summary["inputs"]:
                continue
            summary["rows"].append({k:v for k,v in row.items() if k in
                                   {"variant", "median", "p16", "p84", "b_z", "b_z_err", "b_z_ci95", "b_ell"}})
            summary["inputs"][variant] = f"{name}.npz"
    (data / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--light-curves", type=Path, required=True)
    ap.add_argument("--observed-prior1", type=Path, required=True)
    ap.add_argument("--observed-prior5", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "results/psd_example")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "code"), MPLBACKEND="Agg")
    def run(command):
        command = list(map(str, command))
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    design = args.output / "template_design"
    run([sys.executable, "examples/psd_template_sample.py",
         "--lc-fits", args.light_curves, "--observed-prior1", args.observed_prior1,
         "--observed-prior5", args.observed_prior5, "--output-dir", design])
    paths = {}
    for name, (cohort, variants) in RUNS.items():
        paths[name] = args.output / name
        run([sys.executable, "-m", "framework.psd_steepening",
             "--lc-fits", args.light_curves,
             "--template-manifest", design / "psd_template_manifest.npz",
             "--cohort", cohort, "--output-dir", paths[name], "--variants", variants,
             "--nwalkers", 32, "--nburn", 200, "--nsteps", 500,
             "--step-chunk", 500, "--max-steps", 3000, "--nsamples", 500,
             "--n-jobs", args.jobs, "--mp-start-method", "spawn",
             "--seed", 20261030, "--chain-seed", 0])
    data = args.output / "plot_data"
    print(f"# Reduce fitted PSD results to {data}.", flush=True)
    if args.execute:
        pack_results(paths, data)
    run([sys.executable, "-m", "plotting.plot_psd_steepening", "--data-dir", data,
         "--output", args.output / "psd_steepening.pdf"])


if __name__ == "__main__":
    main()
