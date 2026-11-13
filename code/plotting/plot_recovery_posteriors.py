#!/usr/bin/env python3
"""
Build a small gallery of Stage-1 recovery corner pages for selected sources.

Draw per-source posterior comparisons using the shared diagnostics helpers:
it reuses the same corner plotting routine, but loads only the requested
source rows from each large posterior NPZ and writes one-page PDFs/PNGs plus a
combined multi-page PDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from framework import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/agn-variability-cosmology-mpl-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

_DEFAULT_MPLRC = PROJECT_ROOT / "matplotlibrc"
if _DEFAULT_MPLRC.exists():
    matplotlib.rc_file(str(_DEFAULT_MPLRC))

from matplotlib.backends.backend_pdf import PdfPages

from plotting import diagnostics as cmp


PAPER_PRIOR_COLORS = ["#FF00FF", "#59A14F", "#4C78A8"]


def _split_csv(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    parts = next(csv.reader([str(spec)], skipinitialspace=True))
    parts = [part.strip() for part in parts if part.strip()]
    return parts or None


def _parse_paths(spec: str) -> list[Path]:
    parts = _split_csv(spec)
    if not parts:
        raise ValueError("--inputs must contain at least one path")
    return [Path(p).expanduser().resolve() for p in parts]


def _parse_ints(spec: str | None) -> list[int] | None:
    parts = _split_csv(spec)
    if not parts:
        return None
    return [int(p) for p in parts]


def _metadata_path(npz_path: Path) -> Path | None:
    candidate = npz_path.with_name(npz_path.stem + "_metadata.json")
    return candidate if candidate.exists() else None


def _load_metadata(npz_path: Path) -> dict[str, Any] | None:
    path = _metadata_path(npz_path)
    if path is None:
        return None
    return json.loads(path.read_text())


def _detect_band(data: np.lib.npyio.NpzFile, requested: str | None) -> str:
    bands = cmp._detect_bands(data)
    if requested:
        if requested not in bands:
            raise ValueError(f"Band {requested!r} not found; available bands: {bands}")
        return requested
    if len(bands) != 1:
        raise ValueError(f"Multiple bands found {bands}; pass --band")
    return bands[0]


def _rows(data: np.lib.npyio.NpzFile, key: str, indices: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    return np.asarray(data[key], dtype=float)[indices] * float(scale)


def _optional_rows(
    data: np.lib.npyio.NpzFile,
    key: str,
    indices: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    if key not in data.files:
        return np.full((indices.size, 0), fill_value, dtype=float)
    return np.asarray(data[key], dtype=float)[indices]


def _truth_log10(
    data: np.lib.npyio.NpzFile,
    key_base: str,
    suffix: str,
    indices: np.ndarray,
) -> np.ndarray:
    ln_key = f"true_ln_{key_base}{suffix}"
    log10_key = f"true_log10_{key_base}{suffix}"
    if ln_key in data.files:
        return np.asarray(data[ln_key], dtype=float)[indices] / cmp.LN10
    if log10_key in data.files:
        return np.asarray(data[log10_key], dtype=float)[indices]
    return np.full(indices.size, np.nan, dtype=float)


def _truth_array(
    data: np.lib.npyio.NpzFile,
    per_band_key: str,
    shared_key: str,
    indices: np.ndarray,
) -> np.ndarray:
    if per_band_key in data.files:
        return np.asarray(data[per_band_key], dtype=float)[indices]
    if shared_key in data.files:
        return np.asarray(data[shared_key], dtype=float)[indices]
    return np.full(indices.size, np.nan, dtype=float)


def _run_label_from_meta(npz_path: Path, meta: dict[str, Any] | None, label: str | None) -> tuple[str, str]:
    sampling_label = "unknown"
    if isinstance(meta, dict):
        sampling = meta.get("sampling_model", {})
        if isinstance(sampling, dict):
            sampling_label = str(sampling.get("label", "unknown"))
    if label:
        return label, sampling_label
    if sampling_label != "unknown":
        return cmp._format_sampling_label(sampling_label), sampling_label
    return npz_path.parent.name or npz_path.stem, sampling_label


def _source_indices(source_id: np.ndarray, selected_source_ids: np.ndarray) -> np.ndarray:
    lookup = {int(sid): idx for idx, sid in enumerate(np.asarray(source_id, dtype=int))}
    missing = [int(sid) for sid in selected_source_ids if int(sid) not in lookup]
    if missing:
        raise ValueError(f"Requested source_id(s) not present: {missing}")
    return np.asarray([lookup[int(sid)] for sid in selected_source_ids], dtype=int)


def _load_sparse_run(
    npz_path: Path,
    label: str | None,
    band: str | None,
    selected_source_ids: np.ndarray,
) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    detected_band = _detect_band(data, band)
    suffix = cmp._suffix_for_band(detected_band)
    source_id_all = np.asarray(data["source_id"], dtype=int)
    indices = _source_indices(source_id_all, selected_source_ids)
    source_id = source_id_all[indices]
    if not np.array_equal(source_id, selected_source_ids):
        raise ValueError(f"Source selection mismatch for {npz_path}")

    meta = _load_metadata(npz_path)
    run_label, sampling_label = _run_label_from_meta(npz_path, meta, label)

    posterior_log10_D = _rows(data, f"posterior_ln_D{suffix}", indices, scale=1.0 / cmp.LN10)
    posterior_log10_tau = _rows(data, f"posterior_ln_tau{suffix}", indices, scale=1.0 / cmp.LN10)
    posterior = {
        "log10_D": posterior_log10_D,
        "log10_tau": posterior_log10_tau,
        "log10_sigma2": posterior_log10_D + posterior_log10_tau,
        "log10_jitter": _rows(data, f"posterior_ln_jitter{suffix}", indices, scale=1.0 / cmp.LN10),
        "mean_mag": _rows(data, f"posterior_m{suffix}", indices),
        "m_win_cont": _optional_rows(data, f"posterior_m_win_cont{suffix}", indices),
        "m_win": _optional_rows(data, f"posterior_m_win{suffix}", indices),
    }

    true_log10_tau = _truth_log10(data, "tau", suffix, indices)
    true_log10_sigma = _truth_array(data, f"true_log10_sigma{suffix}", "true_log10_sigma", indices)
    true_log10_D = _truth_log10(data, "D", suffix, indices)
    true_log10_sigma2 = _truth_log10(data, "sigma2", suffix, indices)

    primitive = np.isfinite(true_log10_sigma) & np.isfinite(true_log10_tau)
    true_log10_D[primitive] = 2.0 * true_log10_sigma[primitive] - true_log10_tau[primitive]
    true_log10_sigma2[primitive] = 2.0 * true_log10_sigma[primitive]
    missing_sigma2 = ~np.isfinite(true_log10_sigma2)
    true_log10_sigma2[missing_sigma2] = (true_log10_D + true_log10_tau)[missing_sigma2]

    truth = {
        "log10_D": true_log10_D,
        "log10_sigma2": true_log10_sigma2,
        "log10_tau": true_log10_tau,
        "log10_sigma": true_log10_sigma,
        "mean_mag": _truth_array(data, f"true_mean_mag{suffix}", "true_mean_mag", indices),
        "m_win_cont": _truth_array(data, f"true_m_win_cont{suffix}", "true_m_win_cont", indices),
        "m_win": _truth_array(data, f"true_m_win{suffix}", "true_m_win", indices),
    }

    return {
        "label": run_label,
        "band": detected_band,
        "npz_path": npz_path,
        "meta_path": _metadata_path(npz_path),
        "meta": meta,
        "sampling_label": sampling_label,
        "source_id": source_id,
        "grid_i": np.asarray(data.get("grid_i", np.full(source_id_all.size, -1)), dtype=int)[indices],
        "grid_j": np.asarray(data.get("grid_j", np.full(source_id_all.size, -1)), dtype=int)[indices],
        "posterior": posterior,
        "truth": truth,
        "diag": {},
    }


def _selected_source_ids(first_npz: Path, band: str | None, n_sources: int, seed: int, object_ids: list[int] | None) -> np.ndarray:
    data = np.load(first_npz, allow_pickle=True)
    _detect_band(data, band)
    source_id = np.asarray(data["source_id"], dtype=int)
    if object_ids:
        selected = np.asarray(object_ids, dtype=int)
        _source_indices(source_id, selected)
        return selected
    rng = np.random.default_rng(seed)
    n = min(int(n_sources), source_id.size)
    return np.asarray(source_id[rng.choice(source_id.size, size=n, replace=False)], dtype=int)


def _prior1_logD_mwin_covered_source_ids(first_npz: Path, band: str | None) -> np.ndarray:
    data = np.load(first_npz, allow_pickle=True)
    detected_band = _detect_band(data, band)
    suffix = cmp._suffix_for_band(detected_band)
    source_id = np.asarray(data["source_id"], dtype=int)
    indices = np.arange(source_id.size, dtype=int)

    post_log10_D = np.asarray(data[f"posterior_ln_D{suffix}"], dtype=float) / cmp.LN10
    post_m_win = _optional_rows(data, f"posterior_m_win_cont{suffix}", indices)
    if post_m_win.ndim != 2 or post_m_win.shape[1] == 0:
        raise ValueError(f"Cannot select on m_win coverage: posterior_m_win_cont{suffix} is missing or empty")

    true_log10_tau = _truth_log10(data, "tau", suffix, indices)
    true_log10_sigma = _truth_array(data, f"true_log10_sigma{suffix}", "true_log10_sigma", indices)
    true_log10_D = _truth_log10(data, "D", suffix, indices)
    primitive = np.isfinite(true_log10_sigma) & np.isfinite(true_log10_tau)
    true_log10_D[primitive] = 2.0 * true_log10_sigma[primitive] - true_log10_tau[primitive]
    true_m_win = _truth_array(data, f"true_m_win_cont{suffix}", "true_m_win_cont", indices)

    d16, d84 = np.nanpercentile(post_log10_D, [16.0, 84.0], axis=1)
    m16, m84 = np.nanpercentile(post_m_win, [16.0, 84.0], axis=1)
    covered = (
        np.isfinite(true_log10_D)
        & np.isfinite(true_m_win)
        & np.isfinite(d16)
        & np.isfinite(d84)
        & np.isfinite(m16)
        & np.isfinite(m84)
        & (true_log10_D >= d16)
        & (true_log10_D <= d84)
        & (true_m_win >= m16)
        & (true_m_win <= m84)
    )
    return source_id[covered]


def _selected_covered_source_ids(first_npz: Path, band: str | None, n_sources: int, seed: int) -> np.ndarray:
    eligible = _prior1_logD_mwin_covered_source_ids(first_npz, band)
    if eligible.size < int(n_sources):
        raise ValueError(
            f"Only {eligible.size} source(s) have both true log10D and true m_win within "
            f"Prior 1's 16th-84th posterior interval; requested {int(n_sources)}."
        )
    rng = np.random.default_rng(seed)
    selected = eligible[rng.choice(eligible.size, size=int(n_sources), replace=False)]
    print(
        "Eligible source_ids with Prior 1 truth-in-68% for log10D and m_win: "
        f"{eligible.size}"
    )
    return np.asarray(selected, dtype=int)


def _render_first_page_png(pdf_path: Path, png_path: Path, dpi: int) -> None:
    import fitz

    zoom = float(dpi) / 72.0
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(png_path))
    finally:
        doc.close()


def _write_manifest(
    out_dir: Path,
    runs: list[dict[str, Any]],
    colors: list[str],
    selected_source_ids: np.ndarray,
    combined_pdf: Path,
    selection_note: str | None = None,
) -> None:
    payload = {
        "combined_pdf": str(combined_pdf),
        "selected_source_ids": [int(sid) for sid in selected_source_ids],
        "runs": [
            {
                "label": run["label"],
                "color": color,
                "npz_path": str(run["npz_path"]),
                "sampling_label": run["sampling_label"],
            }
            for run, color in zip(runs, colors)
        ],
    }
    if selection_note:
        payload["selection"] = selection_note
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Corner Gallery", ""]
    lines.append(f"Combined PDF: `{combined_pdf.name}`")
    lines.append("")
    lines.append("## Prior Coding")
    lines.append("")
    for run, color in zip(runs, colors):
        lines.append(f"- `{run['label']}`: `{color}`; `{run['npz_path']}`; `{run['sampling_label']}`")
    lines.append("")
    if selection_note:
        lines.append("## Selection")
        lines.append("")
        lines.append(selection_note)
        lines.append("")
    lines.append("## Source IDs")
    lines.append("")
    lines.append(", ".join(str(int(sid)) for sid in selected_source_ids))
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a corner-plot gallery for random or requested Stage-1 sources.")
    ap.add_argument("--inputs", required=True, help="Comma-separated Stage-1 recovery NPZ paths.")
    ap.add_argument("--labels", default=None, help="Comma-separated run labels, e.g. 'Prior 1,Prior 2,Prior 3'.")
    ap.add_argument("--band", default="Gaia_G", help="Band to plot. Default: Gaia_G.")
    ap.add_argument("--out-dir", required=True, help="Output folder for the gallery.")
    ap.add_argument("--n-sources", type=int, default=10, help="Number of random sources when --object-ids is omitted.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for source selection.")
    ap.add_argument("--object-ids", default=None, help="Optional comma-separated source_id values.")
    ap.add_argument(
        "--select-prior1-covered-logD-mwin",
        action="store_true",
        help="When --object-ids is omitted, select sources where Prior 1's 16th-84th posterior interval contains both true log10(D) and true m_win.",
    )
    ap.add_argument(
        "--presentation",
        action="store_true",
        help="Use the presentation corner layout with log10(D), m, log10(sigma^2), and log10(tau).",
    )
    ap.add_argument("--annotate-run", type=int, default=0, help="0-based run index used for diagonal median annotations.")
    ap.add_argument("--title-note", default="", help="Optional title suffix.")
    ap.add_argument("--detailed-legend", action="store_true", help="Show sampling family/bounds in the legend.")
    ap.add_argument("--no-title", action="store_true", help="Suppress the source title on each page.")
    ap.add_argument("--no-png", action="store_true", help="Do not render per-source PNG previews.")
    ap.add_argument("--png-dpi", type=int, default=120, help="DPI for per-source PNG previews.")
    ap.add_argument("--colors", default=None, help="Optional comma-separated colors, one per input run.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    npz_paths = _parse_paths(args.inputs)
    labels = _split_csv(args.labels)
    if labels is not None and len(labels) != len(npz_paths):
        raise SystemExit("--labels must have the same number of entries as --inputs")
    if args.presentation and labels is None:
        labels = [f"Prior {i + 1}" for i in range(len(npz_paths))]

    object_ids = _parse_ints(args.object_ids)
    selection_note = None
    if object_ids is None and args.select_prior1_covered_logD_mwin:
        selected_source_ids = _selected_covered_source_ids(npz_paths[0], args.band, int(args.n_sources), int(args.seed))
        selection_note = (
            "Selected from sources where Prior 1's 16th--84th posterior interval contains "
            r"both true $\log_{10}(D)$ and true $m_{\rm win}$ (`m_win_cont`)."
        )
    else:
        selected_source_ids = _selected_source_ids(
            npz_paths[0],
            args.band,
            int(args.n_sources),
            int(args.seed),
            object_ids,
        )
    runs = [
        _load_sparse_run(path, labels[i] if labels is not None else None, args.band, selected_source_ids)
        for i, path in enumerate(npz_paths)
    ]
    cmp._validate_compatible_runs(runs)

    if args.presentation:
        corner_params = cmp.PRESENTATION_CORNER_PARAMS
        corner_param_labels = cmp.PRESENTATION_CORNER_PARAM_LABELS
        corner_annotation_math = cmp.PRESENTATION_CORNER_PARAM_ANNOTATION_MATH
        corner_derived_params = {"log10_D", "m_win_cont"}
        corner_style = cmp.CORNER_PRESENTATION_STYLE
    else:
        corner_params = cmp.CORNER_PARAMS
        corner_param_labels = cmp.CORNER_PARAM_LABELS
        corner_annotation_math = cmp.CORNER_PARAM_ANNOTATION_MATH
        corner_derived_params = cmp.DERIVED_CORNER_PARAMS
        corner_style = None

    colors = _split_csv(args.colors)
    if colors is not None and len(colors) != len(runs):
        raise SystemExit("--colors must have the same number of entries as --inputs")
    if colors is not None:
        colors = colors
    elif len(runs) <= len(PAPER_PRIOR_COLORS):
        colors = PAPER_PRIOR_COLORS[: len(runs)]
    else:
        colors = cmp._comparison_palette(len(runs))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf = out_dir / f"corner_gallery_{len(selected_source_ids)}sources.pdf"

    original_legend = cmp._corner_legend_label
    if args.presentation:
        cmp._corner_legend_label = lambda run: cmp._tex_escape_text(run["label"])
    elif not args.detailed_legend:
        cmp._corner_legend_label = lambda run: cmp._tex_escape_text(run["label"])

    try:
        with PdfPages(combined_pdf) as pdf:
            for local_idx, source_id in enumerate(selected_source_ids):
                cmp._add_corner_page(
                    pdf,
                    runs,
                    colors,
                    local_idx,
                    int(source_id),
                    args.title_note,
                    annotate_run_idx=int(args.annotate_run),
                    show_title=not args.no_title,
                    params=corner_params,
                    param_labels=corner_param_labels,
                    annotation_math=corner_annotation_math,
                    derived_params=corner_derived_params,
                    corner_style=corner_style,
                )

        for local_idx, source_id in enumerate(selected_source_ids):
            source_pdf = out_dir / f"source_{int(source_id)}_corner.pdf"
            with PdfPages(source_pdf) as pdf:
                cmp._add_corner_page(
                    pdf,
                    runs,
                    colors,
                    local_idx,
                    int(source_id),
                    args.title_note,
                    annotate_run_idx=int(args.annotate_run),
                    show_title=False,
                    params=corner_params,
                    param_labels=corner_param_labels,
                    annotation_math=corner_annotation_math,
                    derived_params=corner_derived_params,
                    corner_style=corner_style,
                )
            if not args.no_png:
                _render_first_page_png(source_pdf, source_pdf.with_suffix(".png"), int(args.png_dpi))
    finally:
        cmp._corner_legend_label = original_legend

    _write_manifest(out_dir, runs, colors, selected_source_ids, combined_pdf, selection_note)

    print(f"Wrote combined PDF: {combined_pdf}")
    print(f"Wrote per-source files under: {out_dir}")
    print("Selected source_ids:", ",".join(str(int(sid)) for sid in selected_source_ids))


if __name__ == "__main__":
    main()
