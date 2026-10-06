"""Stream the complete labeled Criteo release into immutable split arrays."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 45_840_617
FIELDS = [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def categorical_codes(values, buckets):
    # Criteo categorical strings are published hex hashes, not semantic IDs.
    return np.fromiter((0 if v == '' else 1 + int(v, 16) % (buckets - 1)
                        for v in values), dtype=np.uint32, count=len(values))


def chunks(raw, chunk_size):
    with pd.read_csv(raw, sep='\t', header=None, dtype=str, keep_default_na=False,
                     chunksize=chunk_size) as reader:
        yield from reader


def decode(frame, buckets):
    if frame.shape[1] != 40:
        raise ValueError('Criteo labeled rows must have exactly 40 columns')
    if not frame[0].isin(['0', '1']).all():
        raise ValueError('label must be binary 0/1')
    values = frame.iloc[:, 1:14].replace('', np.nan).to_numpy(dtype=np.float64)
    missing = np.isnan(values)
    if np.isinf(values).any():
        raise ValueError('infinite numerical input')
    values = np.nan_to_num(values, nan=0.0)
    values = np.sign(values) * np.log1p(np.abs(values))
    cats = np.stack([categorical_codes(frame[i].values, buckets) for i in range(14, 40)], axis=1)
    return values, missing.astype(np.float32), cats, frame[0].to_numpy(dtype=np.uint8)


def prepare(raw, output, *, expected_rows=ROWS, buckets=65536, chunk_size=250_000):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if buckets < 2 or expected_rows < 10 or chunk_size < 1:
        raise ValueError('need at least 10 rows, two hash buckets and a positive chunk size')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted preparation never appears as a completed dataset.
    scratch = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        source_hash = sha256(raw)
        cuts = [0, int(expected_rows * .8), int(expected_rows * .9), expected_rows]
        names = ['train', 'validation', 'test']
        arrays = {}
        for name, left, right in zip(names, cuts[:-1], cuts[1:]):
            folder = scratch / name
            folder.mkdir()
            arrays[name] = {key: np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+',
                          dtype=dtype, shape=(right - left, *width))
                for key, dtype, width in [('dense', 'float32', (26,)),
                    ('categorical', 'uint32', (26,)), ('labels', 'uint8', ())]}
        count = 0
        sums, squares = np.zeros(13), np.zeros(13)
        positive = {name: 0 for name in names}
        for frame in chunks(raw, chunk_size):
            dense, missing, cats, labels = decode(frame, buckets)
            end = count + len(frame)
            if end > expected_rows:
                raise ValueError(f'rows exceed expected complete release: {expected_rows}')
            ntrain = max(0, min(end, cuts[1]) - count)
            sums += dense[:ntrain].sum(axis=0)
            squares += np.square(dense[:ntrain]).sum(axis=0)
            joined = np.concatenate([dense, missing], axis=1)
            for name, left, right in zip(names, cuts[:-1], cuts[1:]):
                lo, hi = max(count, left), min(end, right)
                if lo >= hi:
                    continue
                source, target = slice(lo - count, hi - count), slice(lo - left, hi - left)
                for key, value in [('dense', joined), ('categorical', cats), ('labels', labels)]:
                    arrays[name][key][target] = value[source]
                positive[name] += int(labels[source].sum())
            count = end
            print(f'prepared {count:,}/{expected_rows:,} rows', flush=True)
        if count != expected_rows:
            raise ValueError(f'rows {count} differ from expected complete release {expected_rows}')
        mean = sums / cuts[1]
        std = np.sqrt(np.maximum(squares / cuts[1] - mean * mean, 0))
        std[std < 1e-8] = 1
        for name in names:
            dense = arrays[name]['dense']
            for start in range(0, len(dense), chunk_size):
                dense[start:start + chunk_size, :13] = (dense[start:start + chunk_size, :13] - mean) / std
            for value in arrays[name].values():
                value.flush()
        if sha256(raw) != source_hash:
            raise ValueError('source changed during preparation')
        files = {str(path.relative_to(scratch)): sha256(path) for path in sorted(scratch.rglob('*.npy'))}
        spec = {'format_version': 1, 'dataset': 'criteo-display-advertising',
                'source_sha256': source_hash, 'original_rows': count, 'fields': FIELDS,
                'dense_width': 26, 'categorical_width': 26, 'buckets': buckets,
                'dense_transform': 'signed-log1p, train-only mean/std; 13 missing indicators',
                'dense_mean': mean.tolist(), 'dense_std': std.tolist(),
                'split': 'original-order positional 80/10/10; not verified chronological',
                'split_rows': {name: b - a for name, a, b in zip(names, cuts[:-1], cuts[1:])},
                'positives': positive, 'files': files}
        spec['dataset_digest'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        (scratch / 'manifest.json').write_text(json.dumps(spec, indent=2) + '\n')
        os.rename(scratch, output)
        return spec
    except BaseException:
        shutil.rmtree(scratch)
        raise


def load_manifest(path, *, verify=False):
    root = Path(path)
    result = json.loads((root / 'manifest.json').read_text())
    identity = {k: v for k, v in result.items() if k != 'dataset_digest'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != result['dataset_digest']:
        raise ValueError('dataset manifest identity mismatch')
    if verify:
        for relative, expected in result['files'].items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('invalid dataset artifact path')
            if sha256(root / relative) != expected:
                raise ValueError(f'dataset artifact changed: {relative}')
    return result
