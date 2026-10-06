import tempfile
import unittest
from pathlib import Path

from ctrevo.execution import validate_candidate, docker_command


class ExecutionTests(unittest.TestCase):
    def test_native_code_contract_and_training_budget(self):
        source = 'import torch\ndef build_model(schema, config):\n return torch.nn.Linear(2, 1)\n'
        config = validate_candidate({'source': source, 'config': {'lr': .001}}, max_epochs=1)
        self.assertEqual(config['epochs'], 1)
        for bad in ('import os\n' + source, 'x = 1', source.replace('import torch', 'from .. import foo')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_candidate({'source': bad, 'config': {}}, max_epochs=1)
        with self.assertRaises(ValueError):
            validate_candidate({'source': source, 'config': {'epochs': 2}}, max_epochs=1)

    def test_mounts_do_not_expose_holdout_labels_or_credentials(self):
        command = docker_command(image='local-image', task_dir=Path('/data'),
            trial=Path('/trial'), source=Path('/source'), models=Path('/models'),
            venv=Path('/venv'), split='validation')
        joined = ' '.join(command)
        self.assertIn('--gpus all', joined)
        self.assertIn('--network none', joined)
        self.assertIn('/data/train,dst=/train,readonly', joined)
        self.assertIn('/data/validation/dense.npy', joined)
        self.assertNotIn('/data/validation/labels.npy', joined)
        self.assertNotIn('API_KEY', joined)
        self.assertNotIn('src=/data,dst=', joined)


class GpuTests(unittest.TestCase):
    def test_seed_backward_and_logits(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('requires target GPU')
        from ctrevo.seed import build_model
        model = build_model({'dense_width': 26, 'categorical_width': 26, 'buckets': 32}, {}).cuda()
        dense = torch.randn(8, 26, device='cuda')
        cat = torch.randint(0, 32, (8, 26), device='cuda')
        output = model(dense, cat)
        self.assertEqual(tuple(output.shape), (8,))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
