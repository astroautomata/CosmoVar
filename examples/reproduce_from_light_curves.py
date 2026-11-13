#!/usr/bin/env python3
"""Plan or run the Gaia–SDSS figure workflow from the repository root.

- Default: print the plan; add --execute to run it.
- Input FITS: source_id, z and vectors g_transit_time, g_transit_flux,
  g_transit_flux_error, g_transit_mag and variability_flag_g_reject.
- Catalogue LOGL3000: read from the FITS or --catalog-file.
  SDSS-name catalogues also need --crossmatch-file.
- Exclusions: --veto-fits accepts source IDs, including
  assets/selections/gaia_clagn_veto.fits when required.
- Steps: prepare, observed, select, recovery, population, calibration,
  closure, figures, psd.
- Outputs: data/, figures/ and psd_steepening/ under --output.
  Figure 1's renderer updates the bundled graphical model.
- Figure 3: --recovery-input-fits must use the numerical-validity pool
  before the final epoch mask and CLAGN veto to reproduce the original
  template draw; a cleaned pool changes it.
  Paper curves: simulated_light_curves.tar.gz.
- Fresh fits now thin Stage 1 chains along the time axis before pooling
  walkers, correcting the earlier flatten-then-thin procedure, with paired
  Prior 5 mock checks showing median shifts well within per-source posterior
  uncertainties, while archived products remain the reference for exact
  reproduction of the published figures.
- Settings: examples/paper_settings.json.
- Figure data root: COSMOVAR_DATA_ROOT (default: zenodo/).
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from closure_example import read_json, stage1_options

STEPS = ("prepare", "observed", "select", "recovery", "population",
         "calibration", "closure", "figures", "psd")


class Recipe:
    def __init__(self, args):
        self.args = args
        self.data = args.output.resolve() / "data"
        self.figures = args.output.resolve() / "figures"
        self.config = read_json(ROOT / "examples/paper_settings.json")
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / "code"),
                        COSMOVAR_DATA_ROOT=str(self.data), MPLBACKEND="Agg")
        self.observed = self.data / "external_data/observed_gaia_light_curves.fits"
        self.real1 = self.data / "results/observed_light_curves/prior_1/posterior_samples.npz"
        self.real5 = self.data / "results/observed_light_curves/posterior_samples.npz"
        self.quality = self.data / "results/observed_light_curves"
        self.manifest = self.quality / "selected_source_ids.npz"
        self.parent = self.data / "results/cosmology_closure/simulated_population.npz"

    def command(self, command):
        command = list(map(str, command))
        print(shlex.join(command), flush=True)
        if self.args.execute:
            subprocess.run(command, cwd=ROOT, env=self.env, check=True)

    def module(self, name, **options):
        command = [sys.executable, "-m", "framework." + name]
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                command.append(flag)
            elif value is not None and value is not False:
                command.append(f"{flag}={value}")
        self.command(command)

    def fit(self, curves, output, settings):
        if self.args.execute:
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
        self.module("stage1_fitting", input_fits=curves, output_npz=output,
                    n_jobs=self.args.jobs, **stage1_options(settings))

    def prepare(self):
        source = self.data / "external_data/gaia_sdss_input.fits"
        cadence = self.data / "external_data/gaia_g_cadence_pool.fits"
        print(f"# Attach catalogue luminosities; write {source} and {cadence}.", flush=True)
        if self.args.execute:
            import numpy as np
            from astropy.table import Table, Column
            from framework.catalog_utils import load_catalog_z_logl3000
            if source.exists() or cadence.exists() or self.observed.exists():
                raise FileExistsError("Prepared inputs already exist; choose another --output.")
            table = Table.read(self.args.input_fits)
            if "z" not in table.colnames:
                key = next((k for k in ("Z", "redsh") if k in table.colnames), None)
                if key is None:
                    raise ValueError("The matched input needs source_id and redshift.")
                table["z"] = table[key]
            catalogue = self.args.catalog_file or self.args.input_fits
            _, lum_lookup = load_catalog_z_logl3000(
                str(catalogue), crossmatch_file=str(self.args.crossmatch_file) if self.args.crossmatch_file else None)
            table["logL3000"] = np.asarray([lum_lookup.get(int(s), np.nan) for s in table["source_id"]])
            # Epoch rejection changes row lengths; FITS vector columns must allow that.
            for key in list(table.colnames):
                if table[key].ndim > 1:
                    vectors = np.empty(len(table), dtype=object)
                    for i, row in enumerate(table[key]):
                        vectors[i] = np.asarray(row).copy()
                    table.replace_column(key, Column(vectors, name=key, unit=table[key].unit))
            if not np.isfinite(table["logL3000"]).any():
                raise ValueError("No catalogue luminosities matched; check --catalog-file and --crossmatch-file.")
            source.parent.mkdir(parents=True, exist_ok=True)
            table.write(source, overwrite=False)
            if self.args.veto_fits:
                veto = Table.read(self.args.veto_fits)["source_id"]
                table = table[~np.isin(table["source_id"], veto)]
            table.write(cadence, overwrite=False)
        self.module("prepare_flag_filtered_photometry", input_fits=cadence, output_fits=self.observed)

    def observed_fits(self):
        if self.args.execute and (self.data / "results/observed_sample/gaia_sdss_sample.npz").exists():
            raise FileExistsError("Observed summaries already exist; choose another --output.")
        self.module("prepare_observed_sample", lc_gaia_g=self.observed,
                    catalog_file=self.data / "external_data/gaia_sdss_input.fits",
                    output=self.data / "results/observed_sample/gaia_sdss_sample.npz",
                    bands="gaia_g", snr_min=0, mag_max=99, min_epochs_gaia=1, min_points_snr=0)
        for prior, output in ((1, self.real1), (5, self.real5)):
            self.fit(self.observed, output, self.config["observed"][str(prior)])

    def select(self):
        print(f"# Apply observed Prior 1 cuts and write {self.manifest}.", flush=True)
        if not self.args.execute:
            return
        import numpy as np
        from paper_sample_selection import select_observed_prior1
        if self.manifest.exists():
            raise FileExistsError(self.manifest)
        with np.load(self.real1) as d:
            ids = select_observed_prior1(d, self.config["selection"])
        np.savez(self.manifest, source_id_selected=ids)
        settings = {"cuts_submit": dict(self.config["selection"]), "cuts_stage2": {}}
        (self.quality / "inference_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Selected {len(ids):,} Gaia templates.", flush=True)

    def recovery(self):
        curves = self.data / "external_data/prior_sensitivity_light_curves.fits"
        if self.args.execute and curves.exists():
            raise FileExistsError(curves)
        self.module("generate_recovery_grid",
                    input_fits=self.args.recovery_input_fits or self.data / "external_data/gaia_sdss_input.fits",
                    output_fits=curves, tau_days="5,20,80,300,1000,3000,10000",
                    sigma_mag="0.03,0.06,0.10,0.14,0.20,0.30",
                    m0_bin_edges="19.0,19.8,20.3", npts_bin_edges="29,40,52",
                    templates_per_bin=50, min_epochs=10, seed=42, require_full_bins=True)
        for prior in range(1, 6):
            self.fit(curves, self.data / f"results/prior_sensitivity/prior_{prior}/posterior_samples.npz",
                     self.config["recovery"][str(prior)])

    def population(self):
        if self.args.execute and self.parent.exists():
            raise FileExistsError(self.parent)
        import closure_example
        previous = closure_example.run
        closure_example.run = self.module
        try:
            closure_example.generate_population(
                self.config["population"], ROOT, self.parent.parent)
        finally:
            closure_example.run = previous

    def simulate(self, kind):
        folder = "variability_calibration" if kind == "calibration" else "cosmology_closure"
        out = self.data / "results" / folder
        curves = self.data / f"external_data/{folder}_light_curves.fits"
        if self.args.execute and (curves.exists() or (out / "simulated_sources.npz").exists()):
            raise FileExistsError(f"{folder} simulation products already exist.")
        self.module("prepare_closure_simulations", mock_parent_npz=self.parent,
                    mock_parent_schema="fullclosure_truth_parent",
                    real_stage1_posteriors_npz=self.real1, real_used_source_ids_npz=self.manifest,
                    quality_cuts_run_dir=self.quality, cadence_fits=self.observed,
                    out_dir=out, output_fits=curves, sample_label=folder,
                    **self.config["matching"], **self.config["simulation_seeds"][kind])
        if self.args.execute:
            (out / f"{folder}_pairs_and_truth.npz").rename(out / "simulated_sources.npz")
        return out, curves

    def calibration(self):
        out, curves = self.simulate("calibration")
        self.fit(curves, out / "posterior_samples.npz", self.config["calibration"])
        print(f"# Count rest-frame cadence pairs for Figure 5: {out / 'lag_pair_support.npz'}.", flush=True)
        if self.args.execute:
            cadence_support(curves, out / "lag_pair_support.npz")

    def closure(self):
        import closure_example
        out, curves = self.simulate("closure")
        pair = out / "simulated_sources.npz"
        for prior in range(1, 6):
            fit = out / f"prior_{prior}/stage1/posterior_samples.npz"
            self.fit(curves, fit, self.config["closure"][str(prior)])
            if str(prior) not in self.config["stage2"]:
                continue
            settings = self.config["stage2"][str(prior)]
            stage2 = out / f"prior_{prior}/stage2"
            selection = stage2 / "selected_source_ids.npz"
            print(f"# Reapply Prior {prior} Stage 2 cuts; four chains use the same selected sources.", flush=True)
            if self.args.execute:
                import numpy as np
                stage2.mkdir(parents=True, exist_ok=False)
                ids = closure_example.stage2_preselection(fit, pair, settings)
                np.savez(selection, source_id_selected=ids)
                previous = closure_example.run
                closure_example.run = self.module
                try:
                    closure_example.fit_stage2(settings, self.config["population"], ROOT, fit, pair, selection, stage2)
                finally:
                    closure_example.run = previous
            else:
                print(f"# Stage 2 recorded settings: examples/paper_settings.json, stage2.{prior}", flush=True)

    def plots(self):
        import framework
        framework.DATA_ROOT = self.data
        import reproduce_figures
        reproduce_figures.DATA_ROOT = self.data
        for figure, command in reproduce_figures.commands(sys.executable, self.figures).items():
            if figure == 3:
                # The four excluded rows identify failures in the archived fits, not fresh fits.
                command[command.index("--exclude-rows") + 1] = ""
            self.command(command)
        self.command([sys.executable, "-m", "plotting.render_graphical_model"])

    def psd(self):
        self.command([sys.executable, "examples/psd_steepening.py",
                      "--light-curves", self.observed, "--observed-prior1", self.real1,
                      "--observed-prior5", self.real5, "--output", self.args.output / "psd_steepening",
                      "--jobs", self.args.jobs] + (["--execute"] if self.args.execute else []))


def cadence_support(curves, output):
    import numpy as np
    from astropy.io import fits
    values = {key: [] for key in ("source_id", "n_pair_1_31p6", "n_pair_1_100",
                                  "n_adj_0p5_31p6", "n_epoch_time", "g_mag")}
    with fits.open(curves, memmap=True) as hdul:
        for row in hdul[1].data:
            t = np.asarray(row["Gaia_G_transit_time"], float)
            t = np.sort(t[np.isfinite(t)])
            t = (t - t[0]) / (1.0 + row["redsh"])
            delta = np.abs(t[:, None] - t[None, :])[np.triu_indices(len(t), 1)]
            adj = np.diff(t)
            values["source_id"].append(int(row["source_id"]))
            values["n_pair_1_31p6"].append(np.count_nonzero((delta >= 1) & (delta <= 10**1.5)))
            values["n_pair_1_100"].append(np.count_nonzero((delta >= 1) & (delta <= 100)))
            values["n_adj_0p5_31p6"].append(np.count_nonzero((adj >= .5) & (adj <= 10**1.5)))
            values["n_epoch_time"].append(len(t))
            values["g_mag"].append(float(np.nanmedian(row["Gaia_G_mag"])))
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **{k: np.asarray(v) for k,v in values.items()})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-fits", type=Path, required=True)
    ap.add_argument("--catalog-file", type=Path)
    ap.add_argument("--crossmatch-file", type=Path)
    ap.add_argument("--veto-fits", type=Path, help="Optional catalogue-level source_id veto.")
    ap.add_argument("--recovery-input-fits", type=Path, help="Original pre-final-mask cadence pool for an exact recovery-template draw.")
    ap.add_argument("--output", type=Path, default=ROOT / "results/from_light_curves")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="Run the plan. Full fits require HPC resources.")
    args = ap.parse_args()
    recipe = Recipe(args)
    for step in args.steps:
        print(f"\n# {step}", flush=True)
        getattr(recipe, "observed_fits" if step == "observed" else "plots" if step == "figures" else step)()


if __name__ == "__main__":
    main()
