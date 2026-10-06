import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
from ctrevo.agent import CTRAgent

spec = importlib.util.spec_from_file_location('resume_feedback', Path(__file__).parents[1] / 'scripts/resume_with_feedback.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResumeTests(unittest.TestCase):
    def test_feedback_is_carried_once_without_editing_task_or_source(self):
        agent = module.FeedbackAgent('https://example.test', 'private', 'model')
        rejected = {'candidate': {'source': 'original source'}}
        agent.feedback = {'proposal_error': 'exact error', 'rejected_proposal': rejected}
        task = {'task': {'dataset_digest': 'fixed'}}
        with patch.object(CTRAgent, 'propose', return_value={'action': 'stop'}) as propose:
            agent.propose(task)
            self.assertEqual(propose.call_args.args[0]['rejected_proposal'], rejected)
            self.assertEqual(propose.call_args.args[0]['task'], task['task'])
            agent.propose(task)
            self.assertNotIn('proposal_error', propose.call_args.args[0])
        self.assertNotIn('proposal_error', task)
