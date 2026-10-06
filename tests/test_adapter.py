import copy
import tempfile
import unittest
from pathlib import Path

from ctrevo.agent import CTRAgent
from ctrevo.task import CTRTask
from model_evo_harness import load_catalog


class AdapterTests(unittest.TestCase):
    def test_proposal_retry_supplies_concrete_feedback(self):
        class Agent(CTRAgent):
            def _complete(self, instructions, context, effort):
                self.seen.append(copy.deepcopy(context))
                return {'action': 'experiment', 'candidate': {'source': 'bad'}} if len(self.seen) == 1 else {
                    'action': 'stop', 'reason': 'Budget is exhausted'}
        agent = Agent('https://example.test', 'private', 'model')
        agent.seen = []
        actual = agent.propose({'task': {'framework': 'pytorch', 'max_epochs': 1},
                               'catalog': load_catalog(), 'steps': [], 'composition_sources': []})
        self.assertEqual(actual['action'], 'stop')
        self.assertIn('proposal_error', agent.seen[1])
        self.assertIn('rejected_proposal', agent.seen[1])

    def test_task_uses_actual_manifest_and_gpu_contract(self):
        # Unit fixtures validate interfaces only; they are never reported as dataset experiments.
        from ctrevo.data import prepare
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'rows.tsv'
            raw.write_text('\n'.join('\t'.join([str(i % 2)] + ['1'] * 39) for i in range(20)) + '\n')
            prepare(raw, root / 'data', expected_rows=20, buckets=8)
            task = CTRTask(root / 'data', image='image', venv='/venv')
            snapshot = task.snapshot()
            self.assertEqual(snapshot['framework'], 'pytorch')
            self.assertEqual(snapshot['objective']['direction'], 'min')
            self.assertEqual(len(snapshot['fields']), 39)
            self.assertNotIn('event_sequence', snapshot['capabilities'])
            self.assertTrue(snapshot['horizontal_expansion_required'])
            self.assertTrue(task.require_verified_implementation)
            self.assertTrue(snapshot['evaluation_protocol']['require_verified_implementation'])
            self.assertNotIn('test_ctr', str(snapshot))

    def test_host_uses_declared_bindings_and_keeps_attribution_separate(self):
        import hashlib
        import numpy as np
        from unittest.mock import patch
        from ctrevo.data import prepare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'rows.tsv'
            raw.write_text(('\t'.join(['0'] + ['1'] * 39) + '\n') * 20)
            prepare(raw, root / 'data', expected_rows=20, buckets=8)
            task = CTRTask(root / 'data', image='image', venv='/venv')
            candidate = task.seed_candidate
            components = [{'id': 'embedding', 'instance_path': 'CTRModel.embedding'}]
            runtime = {'device': 'cuda', 'rows_seen': 16, 'prediction_rows': 2,
                'implementation_probe': {'batches': 1,
                    'optimizer_parameter_set_stable': True,
                    'source_sha256': hashlib.sha256(candidate['source'].encode()).hexdigest(),
                    'components': [{**components[0], 'status': 'verified', 'targets': [
                        {'path': 'CTRModel.embedding', 'output_summaries': [
                            {'shape': [2, 26, 16], 'mean': 0., 'std': .01, 'max_abs': .04, 'finite': True}]}]}]}}
            with patch('ctrevo.task.execute', return_value=(np.array([.1, .1]), runtime)) as execute:
                trial = root / 'trial_001'
                trial.mkdir()
                result = task.evaluate({'candidate': candidate,
                    'research': {'model_design': {'components': components}}}, trial)
            self.assertEqual(execute.call_args.kwargs['components'], components)
            self.assertEqual(result['implementation_check']['status'], 'verified')
            self.assertEqual(result['change_audit']['status'], 'unverified')
            self.assertEqual(result['evidence'][-1]['id'], 'trial_001.output_scales')
            self.assertEqual(result['evidence'][-1]['value']['probe_batches'], 1)

    def test_finalization_refuses_unverified_selection_in_strict_mode(self):
        import numpy as np
        from unittest.mock import patch
        from ctrevo.data import prepare
        from model_evo_harness import run_search
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'rows.tsv'
            raw.write_text(('\t'.join(['0'] + ['1'] * 39) + '\n') * 20)
            prepare(raw, root / 'data', expected_rows=20, buckets=8)
            task = CTRTask(root / 'data', image='image', venv='/venv')
            with patch('ctrevo.task.execute', return_value=(np.array([.1, .1]), {'device': 'cuda'})):
                run_search(task, object(), output=root / 'run', catalog=load_catalog(), max_steps=0)
            with patch('ctrevo.task.execute', side_effect=AssertionError('must reject before GPU work')):
                with self.assertRaisesRegex(ValueError, 'verified implementation'):
                    task.finalize(root / 'run')
            self.assertFalse((root / 'run/final').exists())

    def test_engine_accepts_baseline_evidence_and_finalization_rejects_protocol_change(self):
        import json
        import numpy as np
        from unittest.mock import patch
        from ctrevo.data import prepare
        from model_evo_harness import run_search
        class StopAgent:
            def propose(self, context):
                self.context = context
                return {'action': 'stop', 'reason': 'Unit contract check complete'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'rows.tsv'
            raw.write_text(('\t'.join(['0'] + ['1'] * 39) + '\n') * 20)
            prepare(raw, root / 'data', expected_rows=20, buckets=8)
            task = CTRTask(root / 'data', image='image', venv='/venv')
            agent = StopAgent()
            with patch('ctrevo.task.execute', return_value=(np.array([.1, .1]), {'device': 'cuda'})):
                state = run_search(task, agent, output=root / 'run', catalog=load_catalog(), max_steps=1)
            self.assertEqual(state['status'], 'stopped')
            self.assertEqual(len(agent.context['interaction_context']['coverage_pairs']), 6)
            self.assertEqual(agent.context['interaction_context']['history'], [])
            self.assertEqual(state['baseline']['evidence'][0]['scope'], 'task')
            task.batch_size = 1
            with self.assertRaisesRegex(ValueError, 'protocol'):
                task.finalize(root / 'run')

    def test_repair_preserves_prior_errors_and_last_parseable_proposal(self):
        class Agent(CTRAgent):
            def _complete(self, instructions, context, effort):
                self.seen.append(copy.deepcopy(context))
                if len(self.seen) == 1:
                    return {'action': 'experiment', 'candidate': {'source': 'bad'}}
                if len(self.seen) == 2:
                    raise ValueError('invalid final JSON')
                return {'action': 'stop', 'reason': 'Interface test'}
        agent = Agent('https://example.test', 'private', 'model')
        agent.seen = []
        agent.propose({'task': {'framework': 'pytorch', 'max_epochs': 1},
                       'catalog': load_catalog(), 'steps': [], 'composition_sources': []})
        self.assertEqual(len(agent.seen[2]['proposal_errors']), 2)
        self.assertEqual(agent.seen[2]['rejected_proposal']['candidate']['source'], 'bad')
