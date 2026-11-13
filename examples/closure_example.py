#!/usr/bin/env python3
"""Reproduce the Figure 6 closure test using the recorded settings.

- Default: regenerate and verify the mocks, then fit Stage 1 under
  Priors 1–5 and Stage 2 under Priors 1, 3, 4 and 5.
- --parent-npz: reuse the saved parent population.
- --mocks-only: generate and verify the mocks without fitting.
- --check-matching: check the saved mock–Gaia pairings without fitting.
- --check-stage2-selection: check the saved selections for specified priors.
- --use-archived-selection: start from the saved Figure 6 source lists
  instead of selecting sources from the fresh fits.

First extract cosmology_closure_inputs.tar.gz,
observed_gaia_light_curves.tar.gz and simulated_light_curves.tar.gz
into zenodo/. Fresh fits do not need archived Stage 1 posteriors.
Archived-selection mode also needs cosmology_closure_stage2.tar.gz;
selection checks need the corresponding Stage 1 and Stage 2 downloads.
Outputs go under --output (default: results/closure_example/).
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    """Read settings and restore quoted infinite bounds."""
    def restore_bounds(values):
        for key, value in values.items():
            if isinstance(value, str) and value in ("Infinity", "-Infinity"):
                values[key] = float(value)
        return values

    return json.loads(path.read_text(), object_hook=restore_bounds)


def run(module, **settings):
    command = [sys.executable, "-m", f"framework.{module}"]
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not None and value is not False:
            command.append(f"{flag}={value}")
    subprocess.run(command, check=True)


def generate_population(settings, assets_root, output):
    t, sed = settings["truth"], settings["sed_setup"]
    run(
        "generate_closure_population",
        output_dir=output, seed=settings["seed"], band=settings["band"],
        z_sampling=settings["z_sampling"], **settings["input_box"],
        chunk_size=settings["analysis_defaults"]["chunk_size"],
        n_steps_mu=settings.get("n_steps_mu", 1024),
        H0=t["H0"], Om0=t["Om0"], w0=t["w0"], wa=t["wa"], D_pivot=t["D_pivot"],
        d_D=t["d_D"], alpha_D=t["alpha_D"], log_sigma_int=t["log_sigma_int"],
        logL_pivot=t["logL3000_pivot"],
        lambda_rf_fiducial_logl3000=t["lambda_rf_fiducial_logl3000"],
        sed_grid=assets_root / sed["sed_grid"], passband_file=assets_root / sed["passband_file"],
        n_integration_wavelength=sed["n_integration_wavelength"],
    )
    return output / "simulated_population.npz"


def rematch_population(settings, saved, ids, z, logl, mwin):
    from framework.prepare_closure_simulations import _fast_greedy_unique_match, _weighted_standardize

    n = len(saved["real_source_id"])
    matching = settings["matching"]
    if (
        settings["target_mode"] != "all"
        or settings["mock_raw_selection"]["enabled"]
        or settings["mock_raw_selection"]["parent_median_mag_cut_applied_before_simulation"]
        or settings["n_mock_candidates_after_cuts"] != len(ids)
        or n != settings["n_real_targets_available"]
    ):
        raise ValueError("Expected the archived Figure 6 matching pool.")
    weights = tuple(matching["weights"][key] for key in ("z", "logl3000", "m_win"))
    features = _weighted_standardize(
        saved["real_z"], saved["real_logl3000_temple"], saved["real_m_win_med"],
        z, logl, mwin, weights=weights,
    )
    rematch, diag = _fast_greedy_unique_match(
        *features, k_initial=matching["shortlist_k_initial"],
        # The Figure 6 run used 1024. The summary omits the increment.
        k_step=matching.get("shortlist_k_step", 1024), k_max=matching["shortlist_k_max"],
    )
    mismatch = int(np.count_nonzero((rematch < 0) | (ids[rematch] != saved["mock_source_id"])))
    print(f"Rematching changed {mismatch:,} of {n:,} pairs.", flush=True)
    print(json.dumps(diag), flush=True)
    return {"n_pairs": n, "rematch_differences": mismatch, "matching": diag}


def check_matching(settings, population, pairs):
    with np.load(population) as parent, np.load(pairs) as saved:
        fields = [parent[key] for key in ("source_id", "z", "logl3000", "true_m_win")]
        report = rematch_population(settings, saved, *fields)
    if report["rematch_differences"]:
        raise ValueError("Rematching did not recover the saved mock–Gaia pairs.")


def simulate_light_curves(settings, population, saved_pairs, cadence, output):
    from framework.prepare_closure_simulations import (
        _draw_truncated_log10_tau,
        _load_gaia_templates_by_source_id,
        _simulate_closure_light_curves,
    )

    with np.load(saved_pairs) as saved, np.load(population) as parent:
        ids, z, logl, mwin = (parent[key] for key in ("source_id", "z", "logl3000", "true_m_win"))
        report = rematch_population(settings, saved, ids, z, logl, mwin)
        n = report["n_pairs"]
        archived = saved["mock_source_id"]

        # Use the recorded pairing for replay so near-ties cannot change the input sample.
        rows = np.searchsorted(ids, archived)
        if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], archived):
            raise ValueError("Archived mock IDs are missing from the regenerated population.")
        np.testing.assert_array_equal(z[rows], saved["mock_z"])
        np.testing.assert_array_equal(logl[rows], saved["mock_logl3000"])
        np.testing.assert_allclose(mwin[rows], saved["mock_m_win_parent_true"], rtol=0, atol=2e-6)
        real = saved["real_source_id"]
        logd = parent["true_log10_d"][rows]
        np.testing.assert_allclose(logd, saved["mock_log10_d_true"], rtol=0, atol=1e-12)
        tau_model = settings["tau_model"]
        logtau = _draw_truncated_log10_tau(
            rng=np.random.default_rng(tau_model["seed"]), n=n, mu=tau_model["log10_mu"],
            sigma=tau_model["log10_sigma"], lo=np.log10(tau_model["tau_min_days"]),
            hi=np.log10(tau_model["tau_max_days"]),
        )
        tau = np.power(10.0, logtau)
        np.testing.assert_allclose(tau, saved["tau_rest_days"], rtol=1e-12, atol=0)
        templates = _load_gaia_templates_by_source_id(
            cadence_fits=cadence, source_ids=real, mock_z=z[rows],
            min_epochs=settings["lightcurve_simulation"]["min_epochs"],
        )
        curves = output / "cosmology_closure_light_curves.fits"
        _simulate_closure_light_curves(
            out_fits=curves, templates=templates, real_source_id=real, mock_source_id=archived,
            mock_z=z[rows], mock_logl3000=logl[rows], mock_mwin_target_true=mwin[rows],
            mock_log10_d_true=logd, tau_days=tau, sigma_drw_mag=np.sqrt(np.power(10.0, logd) * tau),
            rng_lc=np.random.default_rng(settings["lightcurve_seed"]),
            rng_mwin_cont=np.random.default_rng(settings["mwin_cont_seed"]),
            anchor_mode=settings["anchor_mode"],
        )
    return curves, report


def verify_curves(curves, archived):
    from astropy.io import fits

    differences = {}
    with fits.open(curves) as replay, fits.open(archived) as saved:
        a, b = replay[1].data, saved[1].data
        for key in ("source_id", "template_source_id", "true_npts"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        for key in ("Gaia_G_transit_time", "Gaia_G_mag_err", "Gaia_G_mag", "Gaia_G_mag_true"):
            largest = 0.0
            for x, y in zip(a[key], b[key], strict=True):
                if key in ("Gaia_G_transit_time", "Gaia_G_mag_err"):
                    np.testing.assert_array_equal(x, y, err_msg=key)
                else:
                    np.testing.assert_allclose(x, y, rtol=0, atol=2e-6, err_msg=key)
                largest = max(largest, float(np.max(np.abs(x - y))))
            differences[key] = largest
    print(f"Largest absolute differences: {differences}", flush=True)
    return differences


def stage1_options(metadata):
    """Translate a recorded Stage 1 fit into public CLI options."""
    # Read the prior limits from either saved settings format.
    model = metadata["sampling_model"]
    if isinstance(model, str):
        label = model
        bounds = {k: [v["min"], v["max"]] for k, v in metadata["prior_bounds"].items()
                  if v.get("min") is not None}
    else:
        label, bounds = model["label"], model["prior_bounds_used"]
    options = dict(
        bands=",".join(metadata.get("bands_fitted", ["Gaia_G"])), sampling_model=label,
        **{key: metadata[key] for key in (
            "seed", "nwalkers", "nburn", "nsteps", "nsamples_output", "adaptive",
            "target_thinned", "step_chunk", "max_steps", "mwin_mode",
        )},
    )
    for key, flag in (("ln_D", "lnD"), ("ln_tau", "lntau"),
                      ("ln_sigma", "lnsigma"), ("ln_sigma2", "lnsigma2")):
        bound = bounds.get(key)
        if bound is not None:
            options[f"prior_{flag}_min"], options[f"prior_{flag}_max"] = bound
    return options


def fit_stage1(metadata, curves, output, jobs):
    output.parent.mkdir(parents=True, exist_ok=True)
    run("stage1_fitting", input_fits=curves, output_npz=output, n_jobs=jobs,
        **stage1_options(metadata))


def stage2_preselection(stage1, pairs, settings):
    """Apply the recorded posterior-dependent source cuts to fresh joint draws."""
    from framework.stage2_fitting import _mock_catalogue_logl3000

    cuts = settings["cuts_submit"]
    with np.load(stage1) as d, warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        ids, z = d["source_id"], d["redsh"]
        logl = _mock_catalogue_logl3000(pairs, ids, z)
        mag = np.nanmedian(d["posterior_m_win_cont_Gaia_G"], axis=1)
        tau = np.exp(np.nanmedian(d["posterior_ln_tau_Gaia_G"], axis=1))
        npts = d["diag_npts_Gaia_G"]
    keep = (
        np.isfinite(z) & (z >= cuts["z_min"]) & (z <= cuts["z_max"])
        & np.isfinite(logl) & (logl >= cuts["logl_min"]) & (logl <= cuts["logl_max"])
        & np.isfinite(mag) & (mag <= cuts["mag_max_mwin_cont"])
        & np.isfinite(tau) & (tau >= cuts["tau_med_min_days"])
        & np.isfinite(npts) & (npts >= cuts["min_epochs_gaia"])
    )
    return ids[keep]


def fit_stage2(settings, population, assets_root, stage1, pairs, selection, output):
    sampling, cuts = settings["sampling"], settings["cuts_submit"]
    sed = population["sed_setup"]
    priors = {
        f"{param}_{bound}": value
        for param in ("Om0", "w0", "wa")
        for bound, value in zip(("min", "max"), sampling["priors"][param])
    }
    with np.load(selection) as saved:
        requested = saved["source_id_selected"]
    for chain in settings["chains"]:
        posterior = output / f"chain_{chain['chain_id']}.npz"
        if posterior.exists():
            raise FileExistsError(posterior)
        run(
            "stage2_fitting", chains_npz=stage1, chains_npz_original=stage1, pair_npz=pairs,
            source_id_manifest=selection, band=population["band"],
            z_key="redsh", logl3000_key="logL3000",
            m_win_key="posterior_m_win_cont_Gaia_G", lnD_key="posterior_ln_D_Gaia_G",
            sed_grid=assets_root / sed["sed_grid"], passband_file=assets_root / sed["passband_file"],
            n_integration_wavelength=sampling.get("n_integration_wavelength", sed["n_integration_wavelength"]),
            z_min=cuts["z_min"], z_max=cuts["z_max"],
            logl3000_min=cuts["logl_min"], logl3000_max=cuts["logl_max"],
            # The source manifest encodes the paper's posterior-dependent cuts.
            d50_min=settings["cuts_stage2"]["d50_min"], d50_max=settings["cuts_stage2"]["d50_max"],
            draw_stride=sampling["draw_stride"], draw_offset=settings.get("draw_offset", settings.get("nan_handling", {}).get("draw_offset", 0)),
            K=sampling["K_draws"], nan_policy=sampling["nan_policy"],
            min_finite_draws=sampling["min_finite_draws"],
            sample_params=sampling["sample_params"], H0_fixed=sampling["fixed"]["H0"],
            alpha_D_fixed=sampling["fixed"]["alpha_D"], chains=1, seed=chain["seed"],
            chain_method="sequential", output=posterior,
            **priors, **{key: sampling["hyperpriors"][key] for key in (
                "sig_hyp", "sig_alpha", "sig_ell", "logL3000_pivot",
            )},
            **{key: sampling[key] for key in (
                "warmup", "samples", "target_accept_prob", "max_tree_depth", "n_steps_z",
            )},
        )
        with np.load(posterior) as draws:
            fitted = draws["source_id"]
        if not np.isin(fitted, requested).all():
            raise ValueError("Stage 2 returned sources outside the requested manifest.")
        print(f"Chain {chain['chain_id']}: fitted {len(fitted):,} of {len(requested):,} "
              "archived sources after requiring finite paired draws.", flush=True)

    # Save the IDs of sources retained by Stage 2.
    np.savez(selection, source_id_selected=fitted)
    saved_settings = dict(settings)
    saved_settings["chains"] = [
        dict(chain, npz_path=f"chain_{chain['chain_id']}.npz") for chain in settings["chains"]
    ]
    (output / "inference_settings.json").write_text(json.dumps(saved_settings, indent=2) + "\n")


def check_stage2_selection(shared, prior):
    from framework.stage2_fitting import load_mwin_chains

    prior_dir = shared / f"prior_{prior}"
    stage1 = prior_dir / "stage1/posterior_samples.npz"
    stage2 = prior_dir / "stage2"
    pairs = shared / "simulated_sources.npz"
    settings = read_json(stage2 / "inference_settings.json")
    cuts, final, sampling = (settings[key] for key in ("cuts_submit", "cuts_stage2", "sampling"))
    if (
        sampling["nan_policy"] != "finite-paired-draws"
        or np.isfinite(cuts.get("snr_min", -np.inf))
        or np.isfinite(cuts["log10D_half68_max"])
    ):
        raise ValueError("Expected the Figure 6 selection settings.")
    mwin_key = "posterior_m_win_cont_Gaia_G"
    preselected = stage2_preselection(stage1, pairs, settings)
    count = len(preselected)
    if count != settings["submit_cut_summary"]["n_pass"]:
        raise ValueError(f"Prior {prior}: {count:,} sources passed the preliminary cuts, "
                         f"expected {settings['submit_cut_summary']['n_pass']:,}.")
    # Use the fitter's final cuts and joint draw selection.
    with TemporaryDirectory() as scratch:
        quality_ids = Path(scratch) / "quality_source_ids.npz"
        np.savez(quality_ids, source_id_selected=preselected)
        selected, _, _, _, _, metadata = load_mwin_chains(
            chains_npz=stage1, pair_npz=pairs, source_id_manifest=quality_ids,
            band="gaia_g", z_key="redsh", logl3000_key="logL3000",
            m_win_key=mwin_key, ln_d_key="posterior_ln_D_Gaia_G",
            z_min=cuts["z_min"], z_max=cuts["z_max"],
            logl3000_min=cuts["logl_min"], logl3000_max=cuts["logl_max"],
            d50_min=final["d50_min"], d50_max=final["d50_max"], max_objects=final["max_objects"],
            draw_stride=sampling["draw_stride"], draw_offset=settings["nan_handling"]["draw_offset"],
            k_keep=sampling["K_draws"], nan_policy=sampling["nan_policy"],
            min_finite_draws=sampling["min_finite_draws"],
        )
    with np.load(stage2 / "selected_source_ids.npz", allow_pickle=False) as archived:
        np.testing.assert_array_equal(selected, archived["source_id_selected"], err_msg=f"Prior {prior}")
    draws_per_source = metadata["arrays"]["K"]
    if draws_per_source != settings["nan_handling"]["K_use"]:
        raise ValueError(f"Prior {prior}: unexpected number of paired draws per source.")
    print(f"Prior {prior}: recovered all {len(selected):,} saved Stage 2 IDs in order, "
          f"with {draws_per_source} paired draws per source.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=ROOT / "zenodo")
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/closure_example")
    parser.add_argument("--jobs", type=int, default=1, help="Stage 1 workers.")
    parser.add_argument("--priors", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--parent-npz", type=Path, help="Use a saved parent population instead of regenerating it.")
    parser.add_argument("--use-archived-selection", action="store_true",
                        help="Keep the archived source IDs instead of selecting from the fresh fits.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mocks-only", action="store_true", help="Verify the mocks and stop.")
    mode.add_argument("--check-matching", action="store_true", help="Verify archived pairs from the saved parent.")
    mode.add_argument("--check-stage2-selection", nargs="+", type=int, choices=(1, 3, 4, 5),
                      metavar="PRIOR", help="Check archived Stage 2 selections without fitting.")
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    shared = data / "results/cosmology_closure"
    if args.check_stage2_selection:
        for prior in args.check_stage2_selection:
            check_stage2_selection(shared, prior)
        return
    sim_settings = read_json(shared / "simulation_settings.json")
    pair = shared / "simulated_sources.npz"
    if args.check_matching:
        check_matching(sim_settings, args.parent_npz or data / "cosmology_closure_parent_population.npz", pair)
        return
    config = read_json(ROOT / "examples/paper_settings.json")
    pop_settings = read_json(shared / "population_settings.json")
    output.mkdir(parents=True, exist_ok=False)
    population = args.parent_npz or generate_population(pop_settings, args.assets_root.resolve(), output)
    curves, report = simulate_light_curves(
        sim_settings, population, pair, data / "external_data/observed_gaia_light_curves.fits", output,
    )
    report["light_curve_max_abs_difference"] = verify_curves(
        curves, data / "external_data/cosmology_closure_light_curves.fits",
    )
    (output / "mock_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.mocks_only:
        return
    for prior in dict.fromkeys(args.priors):
        prior_dir = output / f"prior_{prior}"
        stage1 = prior_dir / "stage1/posterior_samples.npz"
        fit_stage1(config["closure"][str(prior)], curves, stage1, args.jobs)
        if str(prior) not in config["stage2"]:
            continue
        settings = config["stage2"][str(prior)]
        stage2_dir = prior_dir / "stage2"
        stage2_dir.mkdir(parents=True)
        selection = stage2_dir / "selected_source_ids.npz"
        if args.use_archived_selection:
            with np.load(shared / f"prior_{prior}/stage2/selected_source_ids.npz") as d:
                selected = d["source_id_selected"]
        else:
            selected = stage2_preselection(stage1, pair, settings)
        np.savez(selection, source_id_selected=selected)
        fit_stage2(settings, pop_settings, args.assets_root.resolve(), stage1, pair, selection, stage2_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    os.environ["PYTHONPATH"] = str(ROOT / "code")
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
