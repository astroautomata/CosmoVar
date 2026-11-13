#!/usr/bin/env python3
"""Build the end-to-end closure-test corner plot used in the paper."""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
from pathlib import Path

from plotting import plot_cosmology_posteriors as corner_plot


DEFAULT_OUT_PDF = (
    PROJECT_ROOT
    / 'figures/figure_06_cosmology_closure.pdf'
)
RUN_DIR_TEMPLATE = "results/cosmology_closure/prior_{prior}/stage2"
PRIOR_COLORS = {
    1: "#762A83",
    3: "#FF7F00",
    4: "#7FBC41",
    5: "#D7301F",
}


def _run_specs(repo_root: Path) -> list[tuple[Path, str, str, str]]:
    return [
        (
            repo_root / RUN_DIR_TEMPLATE.format(prior=prior),
            f"Prior {prior}",
            PRIOR_COLORS[prior],
            f"P{prior}",
        )
        for prior in (1, 3, 4, 5)
    ]


def build_v4(
    repo_root: Path = DATA_ROOT,
    out_pdf: Path = DEFAULT_OUT_PDF,
) -> Path:
    """Render the paper closure-test plot and return its output path."""
    repo_root = Path(repo_root).expanduser().resolve()
    out_pdf = Path(out_pdf).expanduser().resolve()
    corner_plot.make_multi_overlay_corner_pdf(
        out_pdf,
        _run_specs(repo_root),
        max_chain_ids=[3, 3, 4, 4],
        annotate_run_idx=3,
        show_group_colors=False,
        quantile_lines_annotated_only=True,
        filled_contours=True,
        smooth_reference_style=True,
        legend_bbox_to_anchor=(0.91, 0.84),
        axis_check_json=out_pdf.with_suffix(".axischeck.json"),
    )
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    args = parser.parse_args()
    build_v4(args.repo_root, args.out_pdf)


if __name__ == "__main__":
    main()
