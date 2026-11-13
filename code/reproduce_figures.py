#!/usr/bin/env python3
"""Reproduce the paper figures and prepare compact plotting data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


from framework import DATA_ROOT, PROJECT_ROOT


CODE_DIR = PROJECT_ROOT / "code"
REPO_ROOT = PROJECT_ROOT
MANIFEST = REPO_ROOT / "zenodo" / 'producer_inputs.tsv'
FIGURES = (2, 3, 4, 5, 6)


def required_inputs(figures: set[int]) -> list[Path]:
    with MANIFEST.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [
            DATA_ROOT / row["source_path"]
            for row in rows
            if figures.intersection(int(value) for value in row["paper_figures"].split(","))
        ]


def figure_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(CODE_DIR), env.get("PYTHONPATH"))))
    return env


def run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def commands(python: str, output: Path, *, reuse_cache: bool = False) -> dict[int, list[str]]:
    closure = str(DATA_ROOT / "results/variability_calibration")
    pair_npz = f"{closure}/simulated_sources.npz"
    mock_stage1 = f"{closure}/posterior_samples.npz"
    pair_support = f"{closure}/lag_pair_support.npz"
    real_stage1 = str(DATA_ROOT / "results/observed_light_curves/posterior_samples.npz")

    selected_commands = {
        2: [
            python,
            "code/plotting/plot_observed_sample.py",
            "--quality-npz",
            str(DATA_ROOT / "results/observed_sample/gaia_sdss_sample.npz"),
            "--output-dir",
            str(output),
            "--output-stem",
            'figure_02_observed_sample',
        ],
        3: [
            python,
            "code/plotting/plot_prior_sensitivity.py",
            "--n-draws-per-source",
            "2000",
            "--exclude-rows",
            "9624,14729,14818,20092",
            "--palette",
            "paperpalette",
            "--left-interpolation",
            "pchip",
            "--left-interp-points",
            "300",
            "--left-smooth-sigma",
            "5",
            "--subtitle-fontsize",
            "10.8",
            "--left-panel-pdf",
            str(output / 'figure_03_prior_sensitivity_left.pdf'),
            "--left-summary-npz",
            str(output / 'figure_03_prior_sensitivity_left.summary.npz'),
            "--right-panel-pdf",
            str(output / 'figure_03_prior_sensitivity_right.pdf'),
            "--output-pdf",
            str(output / 'figure_03_prior_sensitivity.pdf'),
            "--force",
            "--force-summary",
        ],
        4: [
            python,
            "code/plotting/plot_real_vs_simulated.py",
            "--pair-npz",
            pair_npz,
            "--stage1-posteriors-npz",
            mock_stage1,
            "--real-stage1-posteriors-npz",
            real_stage1,
            "--output-pdf",
            str(output / 'figure_04_observed_and_simulated.pdf'),
            "--paper-row-labels",
            "--compact-ell-axis-labels",
            "--compact-z-axis-label",
            "--mwin-med-axis-label",
            "--hide-n-out-labels",
            "--n-label-font-scale",
            "1.05",
            "--axis-label-font-scale",
            "1.14",
            "--tick-label-font-scale",
            "1.14",
            "--colorbar-tick-pad",
            "14",
            "--colorbar-label-pad",
            "22",
            "--column-cmaps",
            "0:plasma,1:plasma,2:viridis,3:viridis",
            "--d-panel-z-range",
            "0.5,3.5",
            "--d-panel-z-bin-width",
            "0.3",
            "--d-binned-panel-xlim=44,47.35",
            "--d-binned-panel-ylim=-5.8,-3.58",
            # Count the broad sample while cropping the first panel.
            "--d-binned-n-label-xlim=43,48",
            "--d-binned-n-label-ylim=-7,-2",
            "--mwin-panel-ylim=14,20.5",
            "--logl-axis-min",
            "44",
            "--no-stage1-summary-cache",
        ],
        5: [
            python,
            "code/plotting/plot_short_lag_calibration.py",
            "--stage1-npz",
            mock_stage1,
            "--pair-npz",
            pair_npz,
            "--pair-support-npz",
            pair_support,
            "--output-prefix",
            str(output / 'figure_05_intermediate'),
            "--paper-only",
            "--paper-output-pdf",
            str(output / 'figure_05_short_lag_calibration.pdf'),
            "--without-retained-cuts",
            "--match-2x4-mock-d-panel-visible-sample",
            "--stage2-plane-cmap",
            "cividis",
            "--stage2-plane-text-side",
            "right",
            "--stage2-plane-sources-wording",
            "--mwin-calib-ylim=-0.03,0.03",
            "--paper-font-scale",
            "1.12",
        ],
        6: [
            python,
            "code/plotting/plot_cosmology_closure.py",
            "--repo-root",
            str(DATA_ROOT),
            "--out-pdf",
            str(output / 'figure_06_cosmology_closure.pdf'),
        ],
    }
    if reuse_cache:
        selected_commands[3] = [arg for arg in selected_commands[3] if arg not in {"--force", "--force-summary"}]
        selected_commands[4].remove("--no-stage1-summary-cache")
    return selected_commands



OUTPUT_NAMES = {2: 'figure_02_observed_sample.pdf', 3: 'figure_03_prior_sensitivity.pdf',
                4: 'figure_04_observed_and_simulated.pdf', 5: 'figure_05_short_lag_calibration.pdf',
                6: 'figure_06_cosmology_closure.pdf'}


def render(bundle, figure, output):
    from plotting.plot_data import load_snapshot, verify_bundle

    verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    out_pdf = output / OUTPUT_NAMES[figure]
    if figure == 2:
        from plotting import plot_observed_sample as plot
        data = load_snapshot(bundle, 'figure_02_observed_sample')
        fig = plot.make_figure(**data)
        plot.save_single_page_pdf(fig, out_pdf)
        plot.plt.close(fig)
    elif figure == 3:
        from plotting import plot_prior_sensitivity as plot
        left = load_snapshot(bundle, 'figure_03_prior_sensitivity_summary')
        right = load_snapshot(bundle, 'figure_03_prior_sensitivity_example')
        args = argparse.Namespace(**left['args'])
        left_pdf, right_pdf = output / 'figure_03_prior_sensitivity_left.pdf', output / 'figure_03_prior_sensitivity_right.pdf'
        plot._plot_left_from_summary(left['summary'], left_pdf, args)
        plot._plot_corner_from_runs(right['runs'], right['colors'], right_pdf, argparse.Namespace(**right['args']))
        plot._compose(left_pdf, right_pdf, out_pdf, args)
    elif figure == 4:
        from plotting.plot_real_vs_simulated import _write_selected_page
        _write_selected_page(out_pdf=out_pdf, **load_snapshot(bundle, 'figure_04_observed_and_simulated'))
    elif figure == 5:
        from plotting import plot_short_lag_calibration as plot
        data = load_snapshot(bundle, 'figure_05_short_lag_calibration')
        plot.plt.rcParams.update(data['rc'])
        plot.make_paper_figure(data['data'], output / 'figure_05_intermediate', paper_output_pdf=out_pdf, **data['kwargs'])
    elif figure == 6:
        from plotting.plot_cosmology_closure import build_v4
        build_v4(bundle, out_pdf)
    else:
        raise ValueError('Unknown figure: ' + str(figure))
    print('Wrote ' + str(out_pdf), flush=True)


def prepare_plot_data(output_dir: Path, cache_dir: Path) -> None:
    """Export figure inputs and saved chains to a verified compact bundle."""
    from plotting.plot_data import SCHEMA_VERSION, sha256, verify_bundle

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError('Output already exists; choose a new empty destination: ' + str(destination))
    missing = [str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p)
               for p in required_inputs({2, 3, 4, 5, 6}) if not p.is_file()]
    if missing:
        raise ValueError('Full figure inputs are required: ' + ', '.join(missing))
    destination.mkdir(parents=True)
    cache = cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = figure_environment()
    env.setdefault('MPLCONFIGDIR', str(cache / '.mplconfig'))
    for figure, command in commands(sys.executable, cache, reuse_cache=True).items():
        if figure != 6:
            subprocess.run(command + ['--export-plot-data', str(destination)], cwd=PROJECT_ROOT, env=env, check=True)
    rows = list(csv.DictReader((PROJECT_ROOT / 'zenodo/producer_inputs.tsv').open(), delimiter='\t'))
    # Include the Stage-2 chains and source selections for source-ID validation.
    for row in rows:
        if '6' in row['paper_figures'].split(','):
            relative = Path(row['source_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid full-data manifest path')
            source, target = DATA_ROOT / relative, destination / relative
            if sha256(source) != row['sha256']:
                raise ValueError('Stage-2 input checksum mismatch: ' + str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    files = {str(f.relative_to(destination)): {'bytes': f.stat().st_size, 'sha256': sha256(f)}
             for f in sorted(destination.rglob('*')) if f.is_file()}
    source_hashes = {f.relative_to(CODE_DIR).as_posix(): sha256(f)
                     for f in sorted(CODE_DIR.rglob('*.py'))}
    manifest = {'schema_version': SCHEMA_VERSION, 'figures': [2, 3, 4, 5, 6], 'files': files,
                'full_input_manifest_sha256': sha256(PROJECT_ROOT / 'zenodo/producer_inputs.tsv'),
                'source_sha256': source_hashes,
                'full_inputs': [{k: row[k] for k in ('source_path', 'sha256', 'paper_figures')} for row in rows],
                'scope': 'Plot-ready arrays and selected posterior draws for the saved-output figures, not a replacement for full posterior inference.'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_bundle(destination)
    print('Verified compact bundle: ' + str(destination), flush=True)
    print('Payload bytes: ' + str(sum(f['bytes'] for f in files.values())), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=FIGURES,
        default=list(FIGURES),
        help="Figures to rebuild (default: 2 3 4 5 6).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse matching plotting summaries and cached panels; rebuild missing or stale caches.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full-inputs", action="store_true", help="Plot from regenerated full producer inputs listed in zenodo/producer_inputs.tsv.")
    mode.add_argument("--plot-data", type=Path, help="Render from a verified compact plotting-data bundle instead of full posterior archives.")
    mode.add_argument("--prepare-plot-data", type=Path, metavar="DIRECTORY",
                      help="Prepare a new compact bundle from the full figure inputs.")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "figures",
                        help="Plot-output directory used when preparing a compact bundle.")
    parser.add_argument("--render-figure", type=int, choices=FIGURES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.prepare_plot_data is not None:
        if args.dry_run:
            parser.error("--dry-run is only available for figure rendering")
        try:
            prepare_plot_data(args.prepare_plot_data, args.cache_dir)
        except ValueError as error:
            parser.error(str(error))
        return
    if args.render_figure is not None:
        if args.plot_data is None:
            parser.error("--render-figure requires --plot-data")
        render(args.plot_data.expanduser().resolve(), args.render_figure,
               args.output_dir.expanduser().resolve())
        return

    if args.plot_data is None and not args.full_inputs:
        args.plot_data = PROJECT_ROOT / "plot_data"

    selected = set(args.figures)
    if args.plot_data is not None:
        from plotting.plot_data import verify_bundle
        bundle = args.plot_data.expanduser().resolve()
        if not (bundle / "manifest.json").is_file():
            parser.error("Compact data not found. Extract compact_plot_data.tar.gz at the repository root, or pass --plot-data DIRECTORY.")
        verify_bundle(bundle)
        output = args.output_dir.expanduser().resolve()
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
        for figure in FIGURES:
            if figure in selected:
                run([args.python, str(CODE_DIR / "reproduce_figures.py"), "--plot-data", str(bundle), "--render-figure", str(figure), "--output-dir", str(output)], env=env, dry_run=args.dry_run)
        return
    missing = [path for path in required_inputs(selected) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}" for path in missing)
        raise SystemExit(
            "Full producer inputs are missing. The Zenodo deposit does not include all "
            "raw Stage-1 producer files for Figures 2–5. Use the compact bundle to "
            "replot the paper figures, or regenerate the inputs listed in "
            f"zenodo/producer_inputs.tsv:\n{formatted}"
        )

    output = args.output_dir.expanduser().resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    env = figure_environment()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))

    all_commands = commands(args.python, output, reuse_cache=args.reuse_cache)
    for figure in FIGURES:
        if figure in selected:
            print(f"\nRebuilding paper Figure {figure}", flush=True)
            run(all_commands[figure], env=env, dry_run=bool(args.dry_run))

    print(f"\nRequested figures complete: {output}", flush=True)


if __name__ == "__main__":
    main()
