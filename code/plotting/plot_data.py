"""Portable plot arrays, stored without Python pickles or local cache identities."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import numpy as np

SCHEMA_VERSION = 1
SNAPSHOTS = {2: ('figure_02_observed_sample',), 3: ('figure_03_prior_sensitivity_summary', 'figure_03_prior_sensitivity_example'), 4: ('figure_04_observed_and_simulated',), 5: ('figure_05_short_lag_calibration',)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(directory, name, data):
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays are not portable plot data')
            key = 'array_' + str(len(arrays))
            arrays[key] = value
            return {'array': key}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            if not all(isinstance(k, (str, int)) for k in value):
                raise TypeError('Plot-data dictionary keys must be strings or integers')
            return {'dict': [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'tuple' if isinstance(value, tuple) else 'list': [encode(v) for v in value]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return {'scalar': value}
        raise TypeError('Unsupported plot-data value: ' + type(value).__name__)

    tree = encode(data)
    metadata = {'schema_version': SCHEMA_VERSION, 'snapshot': name, 'tree': tree}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + '.npz')
    with path.open('wb') as handle:
        np.savez_compressed(handle, metadata_json=json.dumps(metadata), **arrays)
    return path


def load_snapshot(directory, name):
    with np.load(Path(directory) / (name + '.npz'), allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        if metadata['schema_version'] != SCHEMA_VERSION or metadata['snapshot'] != name:
            raise ValueError('Incompatible plot-data snapshot: ' + name)

        def decode(node):
            if len(node) != 1:
                raise ValueError('Invalid plot-data node')
            kind, value = next(iter(node.items()))
            if kind == 'array':
                return archive[value].copy()
            if kind == 'dict':
                return {decode(k): decode(v) for k, v in value}
            if kind in {'list', 'tuple'}:
                values = [decode(v) for v in value]
                return tuple(values) if kind == 'tuple' else values
            if kind == 'scalar' and (value is None or isinstance(value, (str, int, float, bool))):
                return value
            raise ValueError('Invalid plot-data node type: ' + kind)
        return decode(metadata['tree'])


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema_version'] != SCHEMA_VERSION or manifest['figures'] != [2, 3, 4, 5, 6]:
        raise ValueError('Unsupported plotting-data bundle')
    files = manifest['files']
    required = {name + '.npz' for names in SNAPSHOTS.values() for name in names}
    if not required.issubset(files) or not any(n.startswith('results/') for n in files):
        raise ValueError('Plotting-data manifest is incomplete')
    for name, record in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('Invalid bundle path: ' + name)
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Bundle path escapes its directory: ' + name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Plotting-data checksum mismatch: ' + name)
    return manifest
