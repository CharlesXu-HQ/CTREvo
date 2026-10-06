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
            self.assertNotIn('test_ctr', str(snapshot))

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
            self.assertEqual(state['baseline']['evidence'][0]['scope'], 'task')
            task.batch_size = 1
            with self.assertRaisesRegex(ValueError, 'protocol'):
                task.finalize(root / 'run')
