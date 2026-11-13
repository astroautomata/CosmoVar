"""Sample preparation, simulation and inference."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COSMOVAR_DATA_ROOT", PROJECT_ROOT / "zenodo")).expanduser().resolve()
