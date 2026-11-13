"""Render Figure 1 from its TikZ source as a vector PDF and SVG.

Requires the pinned Python environment and a LaTeX installation with TikZ
and the standalone document class.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile

import fitz

from framework import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT_ROOT / 'figures')
    args = parser.parse_args()
    source = (PROJECT_ROOT / 'figures' / 'graphical_model.tex').read_text()
    start = source.index(r'\begin{tikzpicture}')
    end = source.index(r'\end{tikzpicture}', start) + len(r'\end{tikzpicture}')
    picture = source[start:end]
    document = r'''\documentclass[11pt,border=6pt]{standalone}
\usepackage{amsmath,bm,graphicx,xcolor,tikz}
\usetikzlibrary{arrows.meta}
\definecolor{obsfill}{HTML}{ECECEC}
\definecolor{stageoneorange}{HTML}{00A6D6}
\definecolor{stagetwoblue}{HTML}{C2185B}
\begin{document}
''' + picture + '\n' + r'\end{document}' + '\n'
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='agn-graphical-model-') as directory:
        work = Path(directory)
        (work / 'figure.tex').write_text(document)
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'],
                                cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Figure 1 compilation failed:\n' + result.stdout[-6000:])
        with fitz.open(work / 'figure.pdf') as pdf:
            if len(pdf) != 1:
                raise ValueError('Figure 1 must occupy one page')
            pdf.save(output / 'graphical_model.pdf')
            # Give the README a full-width image while retaining vector labels.
            (output / 'graphical_model.svg').write_text(
                pdf[0].get_svg_image(matrix=fitz.Matrix(2, 2), text_as_path=True))
    print('Wrote ' + str(output / 'graphical_model.pdf'))
    print('Wrote ' + str(output / 'graphical_model.svg'))


if __name__ == '__main__':
    main()
