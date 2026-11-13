#!/usr/bin/env python3
"""The script has three uses:

- Default: read the archived figure tables and save lists of the sources
  used in Figures 3-5.
- --observed-stage1: check that applying the selection cuts to
  the original Prior 1 fit results reproduces the Figure 4 sample.
- --prepare-observed-sample: create the observed Gaia sample by requiring
  at least ten G-band observations and excluding the listed changing-look AGN.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def select_observed_prior1(data, cuts):
    """Select observed templates using the configured Prior 1 cuts."""
    required = ("z_min", "z_max", "mag_max", "min_epochs_gaia", "min_finite_draws")
    missing = set(required) - cuts.keys()
    if missing:
        raise ValueError(f"Missing selection settings: {', '.join(sorted(missing))}.")
    for key in required[:3]:
        value = cuts[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or np.isnan(value):
            raise ValueError(f"Selection setting {key} must be a number, not NaN.")
    if cuts["z_min"] > cuts["z_max"]:
        raise ValueError("Selection z_min must not exceed z_max.")
    for key in required[3:]:
        if isinstance(cuts[key], bool) or not isinstance(cuts[key], int) or cuts[key] < 1:
            raise ValueError(f"Selection setting {key} must be a positive integer.")
    z = data["redsh"]
    logl = data["logL3000"]
    mwin = data["posterior_m_win_cont_Gaia_G"]
    lnd = data["posterior_ln_D_Gaia_G"]
    if mwin.ndim != 2 or mwin.shape != lnd.shape:
        raise ValueError("Expected paired magnitude and rate arrays of shape (sources, draws).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(mwin, axis=1)
    keep = (
        np.isfinite(z) & np.isfinite(logl) & np.isfinite(median)
        & (z >= cuts["z_min"]) & (z <= cuts["z_max"]) & (median <= cuts["mag_max"])
        & (data["diag_npts_Gaia_G"] >= cuts["min_epochs_gaia"])
        # Require the configured minimum of usable paired posterior draws.
        & ((np.isfinite(mwin) & np.isfinite(lnd)).sum(axis=1) >= cuts["min_finite_draws"])
    )
    return np.asarray(data["source_id"], dtype=np.int64)[keep]


def write_observed_analysis_sample(data, output):
    """Apply the paper's source cuts to the deposited, epoch-cleaned parent."""
    from astropy.table import Table

    destination = output / "observed_gaia_analysis_light_curves.fits"
    if destination.exists():
        raise FileExistsError(destination)
    parent = Table.read(data / "external_data/observed_gaia_light_curves.fits")
    veto = Table.read(ROOT / "assets/selections/gaia_clagn_veto.fits")["source_id"]
    n_epoch = np.asarray([len(t) for t in parent["g_transit_time"]])
    keep = (n_epoch >= 10) & ~np.isin(parent["source_id"], veto)
    output.mkdir(parents=True, exist_ok=True)
    parent[keep].write(destination, overwrite=False)
    print(f"Retained {int(keep.sum()):,} of {len(parent):,} observed sources: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=ROOT / "zenodo")
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_sample_selection")
    parser.add_argument("--observed-stage1", type=Path,
                        help="Original observed Prior 1 NPZ: reapply cuts and check saved IDs.")
    parser.add_argument("--prepare-observed-sample", action="store_true",
                        help="Write the observed sample after the >=10-epoch cut and CLAGN veto.")
    args = parser.parse_args()
    if args.prepare_observed_sample:
        write_observed_analysis_sample(args.data, args.output)
        return
    tables = args.data / "results/figure_analysis"
    selections = {}
    with fits.open(args.data / "external_data/prior_sensitivity_light_curves.fits", memmap=True) as h:
        rows = h[1].data
        if len(rows) != 33600:
            raise ValueError("Expected the 33,600 paper recovery light curves.")
        # Read the saved bin assignments; the released Gaia pool has newer epoch cleaning.
        templates = np.unique(rows["template_source_id"])
        if len(templates) != 800:
            raise ValueError("Expected 800 distinct recovery templates.")
        for m in range(4):
            for n in range(4):
                cell = (rows["template_m0_bin"] == m) & (rows["template_npts_bin"] == n)
                if len(np.unique(rows["template_source_id"][cell])) != 50:
                    raise ValueError(f"Expected 50 templates in recovery bin ({m}, {n}).")
        selections["figure_03_template_ids"] = templates
        selections["figure_03_mock_ids"] = np.asarray(rows["source_id"], dtype=np.int64)
    with np.load(tables / "figure_03_prior_1.npz") as d:
        np.testing.assert_array_equal(d["source_id"], selections["figure_03_mock_ids"])
        selections["figure_03_plotted_mock_ids"] = d["source_id"][d["included_in_paper"]]
    with np.load(tables / "figure_02_observed.npz") as d:
        observed = d["source_id"]
    with np.load(tables / "figure_04_observed.npz") as d:
        ids = d["source_id"]
        if len(ids) != 113138 or not np.isin(ids, observed).all():
            raise ValueError("Expected 113,138 Figure 4 sources from the Figure 2 sample.")
        selections["figure_04_observed_ids"] = ids
        selections["figure_04_plotted_observed_ids"] = ids[d["included_in_comparison"]]
    with np.load(tables / "figure_04_simulated.npz") as d:
        selections["figure_04_calibration_mock_ids"] = d["source_id"]
        selections["figure_04_plotted_mock_ids"] = d["source_id"][d["included_in_comparison"]]
    with np.load(tables / "figure_05_calibration.npz") as d:
        np.testing.assert_array_equal(d["source_id"], selections["figure_04_calibration_mock_ids"])
        selections["figure_05_plotted_mock_ids"] = d["source_id"][d["included_in_paper"]]
    if args.observed_stage1:
        from closure_example import read_json

        cuts = read_json(ROOT / "examples/paper_settings.json")["selection"]
        with np.load(args.observed_stage1) as d:
            selected = select_observed_prior1(d, cuts)
        np.testing.assert_array_equal(
            np.sort(selected), np.sort(selections["figure_04_observed_ids"]),
            err_msg="The configured cuts and observed fit do not reproduce the archived Figure 4 cohort.",
        )
        print("Observed Prior 1 cuts reproduce the Figure 4 cohort.")
    args.output.mkdir(parents=True, exist_ok=False)
    # These settings describe the archived cohort, not a fresh selection.
    settings = {
        "cuts_submit": {
            "z_min": 0.5, "z_max": 3.5, "mag_max": 20.0,
            "min_epochs_gaia": 10, "snr_min": 0.0,
        },
        "cuts_stage2": {},
    }
    (args.output / "inference_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    for name, ids in selections.items():
        np.savez_compressed(args.output / f"{name}.npz", source_id_selected=ids)
        print(f"{name}: {len(ids):,} sources")


if __name__ == "__main__":
    main()
