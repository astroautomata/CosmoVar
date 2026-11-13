#!/usr/bin/env python
"""
catalog_utils.py

Shared catalog loading utilities used across analysis and fitting scripts.

Primary use-case: load per-source redshift (z) and LOGL3000, supporting two
catalog styles:
  1) Direct catalogs with 'source_id' plus z and LOGL3000 columns
  2) SDSS catalogs with 'SDSS_NAME' plus z and LOGL3000 columns, mapped to
     'source_id' using a crossmatch FITS.
"""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import os
from typing import Dict, Tuple, Optional

import numpy as np
from astropy.io import fits


SCRIPT_DIR = str(PROJECT_ROOT)

# Shared catalogue defaults for the structure-function analysis
DEFAULT_CATALOG = os.path.join(str(DATA_ROOT), 'external_data/sdss_quasar_catalogue.fits')
DEFAULT_CROSSMATCH = os.path.join(str(DATA_ROOT), 'external_data/gaia_sdss_crossmatch.fits')


def _detect_first_existing_column(col_names, candidates):
    for name in candidates:
        if name in col_names:
            return name
    return None


def load_catalog(catalog_file: str, crossmatch_file: Optional[str] = None) -> Dict[int, float]:
    """
    Load catalog with LOGL3000 only.

    Returns
    -------
    dict : mapping source_id -> LOGL3000
    """
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
        col_names = [c.name for c in hdul[1].columns]

        logl3000_col = _detect_first_existing_column(
            col_names, ["LOGL3000", "logl3000", "logL3000", "LogL3000"]
        )
        if logl3000_col is None:
            raise ValueError(f"LOGL3000 column not found in {catalog_file}; columns={col_names}")
        logl3000 = data[logl3000_col]

        if "source_id" in col_names:
            source_ids = data["source_id"]
            return {int(sid): float(l3000) for sid, l3000 in zip(source_ids, logl3000) if np.isfinite(l3000)}

        if "SDSS_NAME" in col_names or "sdss_name" in col_names:
            sdss_col = "SDSS_NAME" if "SDSS_NAME" in col_names else "sdss_name"
            sdss_names = data[sdss_col]
            sdss_to_logl = {name: l3000 for name, l3000 in zip(sdss_names, logl3000) if np.isfinite(l3000)}

            if crossmatch_file is None:
                crossmatch_file = DEFAULT_CROSSMATCH

            with fits.open(crossmatch_file) as xhdul:
                xdata = xhdul[1].data
                x_source_ids = xdata["source_id"]
                x_sdss_names = xdata["sdss_name"]

            result = {}
            for sid, sname in zip(x_source_ids, x_sdss_names):
                if sname in sdss_to_logl:
                    result[int(sid)] = float(sdss_to_logl[sname])
            return result

        raise ValueError(f"Cannot find source_id or SDSS_NAME in {catalog_file}; columns={col_names}")


def load_catalog_z_logl3000(
    catalog_file: str, crossmatch_file: Optional[str] = None
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Load catalog with both redshift and LOGL3000.

    Supports:
      1) Direct catalogs: 'source_id' + z column + LOGL3000 column
      2) SDSS catalogs: SDSS_NAME + z + LOGL3000, mapped via crossmatch FITS

    Returns
    -------
    (z_dict, logl3000_dict) : tuple(dict, dict)
        Mappings source_id -> z, and source_id -> LOGL3000
    """
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
        col_names = [c.name for c in hdul[1].columns]

        logl3000_col = _detect_first_existing_column(
            col_names, ["LOGL3000", "logl3000", "logL3000", "LogL3000"]
        )
        if logl3000_col is None:
            raise ValueError(f"LOGL3000 column not found in {catalog_file}; columns={col_names}")

        z_col = _detect_first_existing_column(
            col_names, ["z", "Z", "Z_SYS", "Z_SYS_SDSS", "redshift", "REDSHIFT", "redsh"]
        )
        if z_col is None:
            raise ValueError(f"Redshift column not found in {catalog_file}; columns={col_names}")

        logl3000 = data[logl3000_col]
        zvals = data[z_col]

        if "source_id" in col_names:
            source_ids = data["source_id"]
            z_dict = {int(sid): float(z) for sid, z in zip(source_ids, zvals) if np.isfinite(z)}
            logl_dict = {int(sid): float(l) for sid, l in zip(source_ids, logl3000) if np.isfinite(l)}
            return z_dict, logl_dict

        if "SDSS_NAME" in col_names or "sdss_name" in col_names:
            sdss_col = "SDSS_NAME" if "SDSS_NAME" in col_names else "sdss_name"
            sdss_names = data[sdss_col]
            sdss_to_logl = {name: l for name, l in zip(sdss_names, logl3000) if np.isfinite(l)}
            sdss_to_z = {name: z for name, z in zip(sdss_names, zvals) if np.isfinite(z)}

            if crossmatch_file is None:
                crossmatch_file = DEFAULT_CROSSMATCH

            with fits.open(crossmatch_file) as xhdul:
                xdata = xhdul[1].data
                x_source_ids = xdata["source_id"]
                x_sdss_names = xdata["sdss_name"]

            z_dict = {}
            logl_dict = {}
            for sid, sname in zip(x_source_ids, x_sdss_names):
                if sname in sdss_to_z:
                    z_dict[int(sid)] = float(sdss_to_z[sname])
                if sname in sdss_to_logl:
                    logl_dict[int(sid)] = float(sdss_to_logl[sname])
            return z_dict, logl_dict

        raise ValueError(f"Cannot find source_id or SDSS_NAME in {catalog_file}; columns={col_names}")
