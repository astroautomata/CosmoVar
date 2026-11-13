#!/usr/bin/env python3
"""
Build a full-sample Gaia G light-curve FITS with variability_flag_g_reject
epochs removed, for all 215,985 flag-cleaned sources.

Applies the same base validity mask used in stage1_fitting.py (finite values,
positive flux/flux_error/mag_error), then additionally removes epochs where
variability_flag_g_reject is True.

The output FITS removes rejected epochs from all per-epoch columns. It supplies
the observed light curves fitted in Figure 4 and summarised in Figure 2.
"""

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from astropy.table import Table

SCRIPT_DIR = str(PROJECT_ROOT)
FACTOR_MAGERR = 2.5 / np.log(10.0)


def _valid_and_varreject_keep(row):
    time     = np.asarray(row["g_transit_time"],       dtype=float)
    mag      = np.asarray(row["g_transit_mag"],        dtype=float)
    flux     = np.asarray(row["g_transit_flux"],       dtype=float)
    flux_err = np.asarray(row["g_transit_flux_error"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = FACTOR_MAGERR * (flux_err / flux)

    base = (
        np.isfinite(time) & np.isfinite(mag)
        & np.isfinite(flux) & np.isfinite(flux_err)
        & np.isfinite(mag_err)
        & (flux > 0.0) & (flux_err > 0.0) & (mag_err > 0.0)
    )
    var_reject = np.asarray(row["variability_flag_g_reject"], dtype=bool)
    keep = base & ~var_reject
    return base, var_reject, keep


def _vector_col_names(tbl):
    return [
        col for col in tbl.colnames
        if hasattr(tbl[col][0], "__len__") or tbl[col].dtype == object
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-fits",
        default=os.path.join(
            str(DATA_ROOT),
            "external_data/gaia_g_cadence_pool.fits",
        ),
        help="Flag-cleaned FITS with variability_flag_g_reject column.",
    )
    parser.add_argument(
        "--output-fits",
        required=True,
        help="Output masked FITS path.",
    )
    parser.add_argument(
        "--cutflow-json",
        default=None,
        help="Optional cutflow JSON. Default: <output-fits>.cutflow.json",
    )
    parser.add_argument(
        "--min-points", type=int, default=10,
        help="Minimum valid epochs after masking (used only for reporting).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cutflow_json is None:
        args.cutflow_json = args.output_fits.replace(".fits", "_cutflow.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_fits)), exist_ok=True)

    if os.path.exists(args.output_fits) and not args.overwrite:
        raise FileExistsError(f"{args.output_fits} already exists; pass --overwrite to replace.")

    print(f"Reading input FITS: {args.input_fits}", flush=True)
    tbl = Table.read(args.input_fits)
    n_sources = len(tbl)
    print(f"  {n_sources:,} sources", flush=True)

    # Identify vector columns (variable-length arrays per row)
    vector_cols = _vector_col_names(tbl)
    n_time_col = "g_transit_time"
    print(f"  Vector columns: {vector_cols}", flush=True)

    # Build output table (start as copy, then overwrite vector columns in place)
    out = tbl.copy()

    total_raw       = 0
    total_base      = 0
    total_varreject = 0
    total_keep      = 0
    n_below_min     = 0

    for i in range(n_sources):
        row = tbl[i]
        base, var_reject, keep = _valid_and_varreject_keep(row)

        n_raw  = len(np.asarray(row[n_time_col]))
        n_base = int(base.sum())
        n_rej  = int((base & var_reject).sum())
        n_kp   = int(keep.sum())

        total_raw       += n_raw
        total_base      += n_base
        total_varreject += n_rej
        total_keep      += n_kp
        if n_kp < args.min_points:
            n_below_min += 1

        # Strip rejected/invalid epochs from every vector column
        for col in vector_cols:
            arr = np.asarray(row[col])
            if arr.size == n_raw:
                out[col][i] = arr[keep]

        if (i + 1) % 25000 == 0:
            print(f"  {i+1:,}/{n_sources:,} processed...", flush=True)

    print(f"\nCutflow summary:", flush=True)
    print(f"  Sources:                             {n_sources:,}", flush=True)
    print(f"  Total raw epochs:                    {total_raw:,}", flush=True)
    print(f"  Valid (base) epochs:                 {total_base:,}", flush=True)
    print(f"  variability_flag_g_reject (in valid):{total_varreject:,}  "
          f"({100.*total_varreject/max(1,total_base):.2f}%)", flush=True)
    print(f"  Valid after flag mask:               {total_keep:,}", flush=True)
    print(f"  Sources with <{args.min_points} valid epochs after mask: {n_below_min:,}", flush=True)

    cutflow = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_fits": os.path.abspath(args.input_fits),
        "output_fits": os.path.abspath(args.output_fits),
        "flag_policy": {
            "base_validity": (
                "finite g_transit_time, g_transit_mag, g_transit_flux, "
                "g_transit_flux_error, mag_err=2.5/ln10*flux_err/flux > 0"
            ),
            "epoch_mask": "variability_flag_g_reject == False",
        },
        "n_sources": n_sources,
        "min_points": args.min_points,
        "total_raw_epochs": total_raw,
        "total_valid_before_flag_mask": total_base,
        "total_variability_flag_g_reject_within_valid": total_varreject,
        "total_valid_after_flag_mask": total_keep,
        "sources_below_min_points": n_below_min,
    }

    print(f"\nWriting masked FITS: {args.output_fits}", flush=True)
    out.write(args.output_fits, overwrite=args.overwrite)
    print(f"Writing cutflow JSON: {args.cutflow_json}", flush=True)
    with open(args.cutflow_json, "w") as fh:
        json.dump(cutflow, fh, indent=2)
        fh.write("\n")

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
