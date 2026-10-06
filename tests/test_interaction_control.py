import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ctrevo.data import prepare
from ctrevo.task import CTRTask


class InteractionControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        raw = self.root / 'rows.tsv'
        raw.write_text(('\t'.join(['0'] + ['1'] * 39) + '\n') * 20)
        prepare(raw, self.root / 'data', expected_rows=20, buckets=8)
        self.task = CTRTask(self.root / 'data', image='image', venv='/venv')
        self.proposal = {'candidate': {**self.task.seed_candidate, 'config': {'model': {'mixed': True}}},
            'research': {'interaction_plan': {'decision': 'test', 'control': {
                'config_patch': {'model': {'mixed': False}}, 'expected_effect': 'Remove mixed term'}}}}
        self.path = self.root / 'trial_001'
        self.path.mkdir()

    def test_full_same_source_control_records_paired_metrics_without_attribution(self):
        saved = copy.deepcopy(self.proposal)
        runtime = {'device': 'cuda', 'rows_seen': 16, 'prediction_rows': 2}
        with patch('ctrevo.task.execute', side_effect=[(np.array([.1, .1]), runtime),
                                                       (np.array([.2, .2]), runtime)]) as execute:
            result = self.task.evaluate(self.proposal, self.path)
        self.assertEqual(execute.call_count, 2)
        main, control = execute.call_args_list
        self.assertEqual(main.args[0]['source'], control.args[0]['source'])
        self.assertEqual(main.kwargs, control.kwargs)
        self.assertEqual(control.args[0]['config']['model'], {'mixed': False})
        self.assertEqual(control.args[2], self.path / 'control')
        self.assertEqual(self.proposal, saved)
        evidence = result['interaction_control']
        self.assertEqual(evidence['status'], 'completed')
        self.assertLess(evidence['paired_logloss']['upper'], 0)
        self.assertEqual(result['change_audit']['status'], 'unverified')
        self.assertEqual(result['evidence'][-1]['id'], 'trial_001.interaction_control')
        self.assertEqual(json.loads((self.path / 'evaluation.json').read_text()), result)

    def test_noop_or_protocol_control_is_rejected_before_training(self):
        for patch_value in ({'model': {'mixed': True}}, {'lr': .02}, {'model': {}}):
            self.proposal['research']['interaction_plan']['control']['config_patch'] = patch_value
            with patch('ctrevo.task.execute') as execute, self.assertRaises(ValueError):
                self.task.evaluate(self.proposal, self.path)
            execute.assert_not_called()

    def test_failed_control_retains_main_result_and_requests_review(self):
        with patch('ctrevo.task.execute', side_effect=[(np.array([.1, .1]), {}), RuntimeError('failure')]):
            result = self.task.evaluate(self.proposal, self.path)
        self.assertEqual(result['interaction_control']['status'], 'failed')
        self.assertNotIn('paired_logloss', result['interaction_control'])
        self.assertAlmostEqual(result['score'], -np.log(.9))
        self.assertTrue(result['review_required'])

    def test_missing_control_output_retains_main_metrics(self):
        with patch('ctrevo.task.execute', side_effect=[(np.array([.1, .1]), {}), FileNotFoundError('output')]):
            result = self.task.evaluate(self.proposal, self.path)
        self.assertEqual(result['interaction_control']['status'], 'failed')
        self.assertEqual(result['interaction_control']['error_type'], 'FileNotFoundError')
        self.assertIn('score', result)
        self.assertTrue(result['review_required'])

    def test_deferred_plan_does_not_spend_control_budget(self):
        self.proposal['research']['interaction_plan'] = {'decision': 'defer'}
        with patch('ctrevo.task.execute', return_value=(np.array([.1, .1]), {})) as execute:
            result = self.task.evaluate(self.proposal, self.path)
        execute.assert_called_once()
        self.assertNotIn('interaction_control', result)

    def test_snapshot_exposes_source_bound_views_and_budget(self):
        snapshot = self.task.snapshot()
        self.assertTrue(snapshot['interaction_plan_required'])
        views = {v['id']: v for v in snapshot['interaction_views']}
        self.assertEqual(views['numeric']['fields'], views['missing']['fields'])
        self.assertEqual(len(views['categorical']['fields']), 26)
        self.assertEqual(snapshot['evaluation_protocol']['interaction_controls_per_trial'], 1)
