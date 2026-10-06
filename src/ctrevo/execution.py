"""Execute candidate training in Docker with no held-out labels or secrets."""

import ast
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def validate_candidate(candidate, *, max_epochs):
    if not isinstance(candidate, dict) or not isinstance(candidate.get('source'), str):
        raise ValueError('candidate needs Python source and optional config')
    if len(candidate['source']) > 100_000:
        raise ValueError('candidate source exceeds 100000 characters')
    tree = ast.parse(candidate['source'])
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'build_model' for n in tree.body):
        raise ValueError('candidate must export build_model(schema, config)')
    for node in ast.walk(tree):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        for module in modules:
            if (isinstance(node, ast.ImportFrom) and node.level) or not (
                module.split('.')[0] in {'torch', 'numpy', 'math', 'typing', 'collections', '__future__'}
                or module.startswith('model_evo_harness.models.pytorch.')):
                raise ValueError(f'candidate import is not available: {module}')
    config = {'lr': .002, 'weight_decay': 1e-6, 'epochs': 1, 'model': {}}
    supplied = candidate.get('config', {})
    if not isinstance(supplied, dict) or set(supplied) - set(config):
        raise ValueError('config supports lr, weight_decay, epochs, model')
    config.update(supplied)
    if type(config['epochs']) is not int or not 1 <= config['epochs'] <= max_epochs:
        raise ValueError('epochs exceed frozen training budget')
    for key, lower, upper in [('lr', 1e-6, .1), ('weight_decay', 0, .1)]:
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'invalid {key}')
    if not isinstance(config['model'], dict):
        raise ValueError('model configuration must be a dictionary')
    json.dumps(config, allow_nan=False)
    return config


def docker_command(*, image, task_dir, trial, source, models, venv, split, checkpoint=None):
    if split not in ('validation', 'test'):
        raise ValueError('invalid target split')
    command = ['docker', 'run', '--rm', '--gpus', 'all', '--cidfile', str(trial / 'container.id'),
        '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--pids-limit', '256', '--memory', '24g', '--cpus', '8',
        '--user', f'{os.getuid()}:{os.getgid()}', '--tmpfs', '/tmp:rw,nosuid,size=1g']
    mounts = [(task_dir / 'train', '/train', True), (trial / 'input', '/input', True),
              (trial / 'output', '/output', False), (source, '/opt/ctr-src', True),
              (models, '/opt/model-src', True), (venv, '/opt/venv', True)]
    mounts += [(task_dir / split / f'{name}.npy', f'/target/{name}.npy', True)
               for name in ('dense', 'categorical')]
    if checkpoint:
        mounts.append((checkpoint, '/checkpoint/weights.pt', True))
    for src, dst, readonly in mounts:
        command += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
    return command + ['--env', 'PYTHONPATH=/opt/ctr-src:/opt/model-src',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', image, '/opt/venv/bin/python', '-m', 'ctrevo.worker']


def execute(candidate, data, trial, *, image, venv, max_epochs, batch_size, seed,
            timeout=3600, split='validation', checkpoint=None, components=None):
    from .data import load_manifest
    config = validate_candidate(candidate, max_epochs=max_epochs)
    trial, data = Path(trial).resolve(), Path(data).resolve()
    manifest = load_manifest(data)
    trial.mkdir(parents=True, exist_ok=True)
    # Attempt artifacts are immutable. Retry in a new trial, never overwrite evidence.
    (trial / 'input').mkdir()
    (trial / 'output').mkdir()
    (trial / 'input/candidate.py').write_text(candidate['source'])
    (trial / 'input/job.json').write_text(json.dumps({'schema': {
        k: manifest[k] for k in ('dense_width', 'categorical_width', 'buckets')},
        'config': config, 'batch_size': batch_size, 'seed': seed,
        'checkpoint': checkpoint is not None,
        'components': [{key: item.get(key) for key in ('id', 'instance_path')}
                       for item in (components or [])]}))
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='ctrevo-models-') as temporary:
        models = Path(temporary)
        dest = models / 'model_evo_harness/models/pytorch'
        dest.mkdir(parents=True)
        (dest.parent / '__init__.py').write_text('')
        (dest.parent.parent / '__init__.py').write_text('')
        reference = root / 'third_party/model-evo-harness/src/model_evo_harness/models/pytorch'
        if not reference.is_dir():
            raise ValueError('initialize the Harness submodule first')
        for path in reference.glob('*.py'):
            if path.is_symlink():
                raise ValueError('reference source must not be a symlink')
            shutil.copyfile(path, dest / path.name)
        command = docker_command(image=image, task_dir=data, trial=trial, source=root / 'src',
            models=models, venv=Path(venv).resolve(), split=split, checkpoint=checkpoint)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b'') + (error.stderr or b'')
            (trial / 'worker.log').write_text(output.decode(errors='replace') if isinstance(output, bytes) else output)
            cid = trial / 'container.id'
            if cid.exists():
                subprocess.run(['docker', 'rm', '-f', cid.read_text().strip()], capture_output=True, timeout=30)
            raise RuntimeError('GPU candidate exceeded timeout') from None
    (trial / 'worker.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('GPU worker failed; inspect local worker.log')
    runtime = json.loads((trial / 'output/runtime.json').read_text())
    if runtime.get('device') != 'cuda' or (not checkpoint and
            runtime.get('rows_seen') != manifest['split_rows']['train'] * config['epochs']):
        raise ValueError('candidate did not complete full-data CUDA training')
    prediction = np.load(trial / 'output/prediction.npy', allow_pickle=False)
    return prediction, runtime
