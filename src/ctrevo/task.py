"""CTR adapter: frozen data, independent metrics and full-data CUDA evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import FIELDS, load_manifest, sha256
from .audit import audit_execution
from .execution import execute, validate_candidate
from .metrics import evaluate, paired_logloss


class CTRTask:
    def __init__(self, data, *, image, venv, seed=42, max_epochs=1, batch_size=8192, timeout=3600,
                 require_verified_implementation=True):
        self.data = Path(data).resolve()
        self.manifest = load_manifest(self.data, verify=True)
        self.seed, self.max_epochs, self.batch_size = seed, max_epochs, batch_size
        self.require_verified_implementation = require_verified_implementation
        self.settings = dict(image=image, venv=venv, seed=seed, max_epochs=max_epochs,
                             batch_size=batch_size, timeout=timeout)
        self.seed_candidate = {'source': Path(__file__).with_name('seed.py').read_text(), 'config': {}}

    def snapshot(self):
        code = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob('*.py')):
            code.update(path.name.encode())
            code.update(path.read_bytes())
        return {'task_id': 'criteo-ctr', 'dataset_digest': self.manifest['dataset_digest'],
            'stage': 'ranking', 'framework': 'pytorch', 'fields': FIELDS,
            'model_design_required': True, 'horizontal_expansion_required': True,
            'capabilities': ['tabular_features', 'observed_outcome_labels'],
            'objective': {'name': 'validation_logloss', 'direction': 'min'},
            'evaluation_protocol': {'unit': 'impression', 'split': self.manifest['split'],
                'metric': 'binary_logloss', 'host_code_sha256': code.hexdigest(),
                'seed': self.seed, 'max_epochs': self.max_epochs, 'batch_size': self.batch_size,
                'implementation_audit': 'declared_component_execution_v1',
                'require_verified_implementation': self.require_verified_implementation,
                'image': self.settings['image'], 'timeout': self.settings['timeout']},
            'max_epochs': self.max_epochs, 'baseline_candidate': self.seed_candidate,
            'feature_schema': {'dense_fields': FIELDS[:13], 'categorical_fields': FIELDS[13:],
                'dense_width': 26, 'categorical_width': 26, 'buckets': self.manifest['buckets'],
                'dense_layout': 'first 13 signed-log1p standardized values, next 13 missing indicators',
                'categorical_layout': 'field-specific hash buckets, zero is missing; no sequence semantics'},
            'feature_groups': [{'id': 'numeric', 'fields': FIELDS[:13],
                                'rationale': 'Published numeric inputs with unknown business semantics'},
                               {'id': 'categorical', 'fields': FIELDS[13:],
                                'rationale': 'Published categorical inputs, grouped by type only'}],
            'evidence': [{'id': 'dataset.train', 'statement': 'Measured complete training partition',
                'source': 'verified preparation manifest', 'status': 'observed', 'scope': 'task',
                'value': {'rows': self.manifest['split_rows']['train'],
                          'positives': self.manifest['positives']['train'], 'original_fields': 39}}]}

    def baseline(self, path):
        return self._evaluate(self.seed_candidate, Path(path),
                              components=[{'id': 'seed_model', 'instance_path': 'CTRModel'}])

    def evaluate(self, proposal, path):
        components = proposal.get('research', {}).get('model_design', {}).get('components', [])
        return self._evaluate(proposal['candidate'], Path(path), components=components)

    def _evaluate(self, candidate, path, *, components):
        prediction, runtime = execute(candidate, self.data, path, components=components, **self.settings)
        labels = np.load(self.data / 'validation/labels.npy', mmap_mode='r')
        metrics = evaluate(labels, prediction)
        baseline_file = path.parent / 'baseline/output/prediction.npy'
        if path.name != 'baseline' and baseline_file.is_file():
            metrics['paired_vs_baseline'] = paired_logloss(labels, np.load(baseline_file), prediction)
        config = validate_candidate(candidate, max_epochs=self.max_epochs)
        check = audit_execution(candidate['source'], components, runtime,
            train_rows=self.manifest['split_rows']['train'] * config['epochs'],
            prediction_rows=len(labels))
        result = {'score': metrics['logloss'], 'metrics': metrics, 'runtime': runtime,
            'implementation_check': check,
            'change_audit': {'status': 'unverified'},
            'review_required': check['status'] != 'verified' or abs(metrics['mean_prediction'] - metrics['observed_ctr']) > .05,
            'evidence': [{'id': f'{path.name}.metrics', **({} if path.name == 'baseline' else {'trial_id': path.name}),
                'statement': 'Independent host validation metrics from complete fixed split',
                'source': f'{path.name}/evaluation.json', 'status': 'observed', 'scope': 'task' if path.name == 'baseline' else 'trial',
                'value': metrics}]}
        (path / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result

    def finalize(self, run):
        run = Path(run).resolve()
        state = json.loads((run / 'journal.json').read_text())
        if state['status'] != 'completed' and state['status'] != 'stopped':
            raise ValueError('final evaluation requires a completed/stopped search')
        if state['identity']['dataset_digest'] != self.manifest['dataset_digest']:
            raise ValueError('dataset differs from search')
        digest = hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        if state['identity']['task_sha256'] != digest:
            raise ValueError('task or evaluation protocol differs from search')
        if any('reflection' not in step for step in state['steps']):
            raise ValueError('finish pending reflection before final evaluation')
        best = state['best_id']
        selected = state['baseline'] if best == 'baseline' else next(
            s['evaluation'] for s in state['steps'] if s['id'] == best)
        if (self.require_verified_implementation and
                selected.get('implementation_check', {}).get('status') != 'verified'):
            raise ValueError('final evaluation requires a verified implementation for the selection')
        final = run / 'final'
        final.mkdir()  # One final evaluation; failures retain artifacts for inspection.
        candidate = self.seed_candidate if best == 'baseline' else next(
            s['proposal']['candidate'] for s in state['steps'] if s['id'] == best)
        predictions = {}
        for identifier, value in [('baseline', self.seed_candidate), (best, candidate)]:
            if identifier in predictions:
                continue
            checkpoint = run / identifier / 'output/weights.pt'
            predictions[identifier], _ = execute(value, self.data, final / identifier,
                split='test', checkpoint=checkpoint, **self.settings)
        labels = np.load(self.data / 'test/labels.npy', mmap_mode='r')
        report = {'selected_id': best, 'dataset_digest': self.manifest['dataset_digest'],
            'baseline': evaluate(labels, predictions['baseline']),
            'selected': evaluate(labels, predictions[best]),
            'paired_logloss': paired_logloss(labels, predictions['baseline'], predictions[best])}
        (final / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
