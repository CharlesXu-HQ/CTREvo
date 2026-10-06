import hashlib
import unittest

from ctrevo.audit import audit_execution


class AuditTests(unittest.TestCase):
    def runtime(self):
        return {'device': 'cuda', 'rows_seen': 20, 'prediction_rows': 4,
                'implementation_probe': {
                    'source_sha256': hashlib.sha256(b'source').hexdigest(),
                    'components': [{'id': 'branch', 'instance_path': 'Model.branch',
                                    'status': 'verified'}], 'batches': 3,
                    'optimizer_parameter_set_stable': True}}

    def check(self, runtime):
        return audit_execution('source', [{'id': 'branch', 'instance_path': 'Model.branch'}],
                               runtime, train_rows=20, prediction_rows=4)

    def test_actual_execution_can_be_verified_without_attributing_a_gain(self):
        result = self.check(self.runtime())
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['scope'], 'declared_component_execution_v1')
        self.assertIn('mathematical', result['limitations'])
        self.assertIn('attribution', result['limitations'])

    def test_missing_or_incomplete_observation_is_not_verified(self):
        for change in ({'implementation_probe': {}}, {'rows_seen': 19}, {'device': 'cpu'},
                       {'prediction_rows': 3}):
            with self.subTest(change=change):
                runtime = self.runtime()
                runtime.update(change)
                self.assertNotEqual(self.check(runtime)['status'], 'verified')
        runtime = self.runtime()
        runtime['implementation_probe']['components'] = []
        self.assertEqual(self.check(runtime)['status'], 'unverified')
        runtime = self.runtime()
        del runtime['implementation_probe']['optimizer_parameter_set_stable']
        self.assertEqual(self.check(runtime)['status'], 'unverified')

    def test_stale_source_or_component_binding_is_not_verified(self):
        runtime = self.runtime()
        runtime['implementation_probe']['source_sha256'] = 'stale'
        self.assertEqual(self.check(runtime)['status'], 'contradicted')
        runtime = self.runtime()
        runtime['implementation_probe']['components'][0]['instance_path'] = 'Model.other'
        self.assertEqual(self.check(runtime)['status'], 'contradicted')

    def test_component_status_is_preserved(self):
        for status in ('unverified', 'contradicted'):
            runtime = self.runtime()
            runtime['implementation_probe']['components'][0]['status'] = status
            self.assertEqual(self.check(runtime)['status'], status)

    def test_changed_optimizer_parameter_set_blocks_verification(self):
        runtime = self.runtime()
        runtime['implementation_probe']['optimizer_parameter_set_stable'] = False
        self.assertEqual(self.check(runtime)['status'], 'contradicted')


class ComponentProbeTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires target GPU and PyTorch')
        if not torch.cuda.is_available():
            self.skipTest('requires target GPU')
        self.torch = torch

    def run_probe(self, model, bindings, candidate=None):
        from ctrevo.audit import ComponentProbe
        torch = self.torch
        probe = ComponentProbe(model, bindings, candidate=candidate)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.zero_grad(set_to_none=True)
        output = model(torch.ones(4, 2, device='cuda'))
        loss = candidate.training_loss(output, torch.ones_like(output)) if candidate else output.sum()
        loss.backward()
        optimizer.step()
        probe.after_step()
        result = probe.result()
        probe.close()
        return result

    def test_connected_module_and_parameterless_method_are_observed(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = torch.nn.Linear(2, 1)
            def interaction(self, value):
                return value.square()
            def forward(self, value):
                return self.interaction(self.branch(value))
        model = Model().cuda()
        original = model.interaction
        result = self.run_probe(model, [
            {'id': 'branch', 'instance_path': 'Model.branch'},
            {'id': 'interaction', 'instance_path': 'Model.interaction'}])
        self.assertEqual([x['status'] for x in result['components']], ['verified', 'verified'])
        self.assertEqual(model.interaction, original)

    def test_called_but_disconnected_branch_cannot_pass(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.used = torch.nn.Linear(2, 1)
                self.unused = torch.nn.Linear(2, 1)
            def forward(self, value):
                self.unused(value)
                return self.used(value)
        result = self.run_probe(Model().cuda(), [{'id': 'branch', 'instance_path': 'Model.unused'}])
        self.assertEqual(result['components'][0]['status'], 'unverified')
        self.assertGreater(result['components'][0]['targets'][0]['calls'], 0)

    def test_modulelist_children_must_all_be_connected(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
            def forward(self, value):
                for layer in self.layers:
                    value = layer(value)
                return value
        result = self.run_probe(Model().cuda(), [{'id': 'layers', 'instance_path': 'Model.layers'}])
        self.assertEqual(result['components'][0]['status'], 'verified')
        self.assertEqual(len(result['components'][0]['targets']), 2)

    def test_missing_declared_path_is_contradicted(self):
        model = self.torch.nn.Linear(2, 1).cuda()
        result = self.run_probe(model, [{'id': 'missing', 'instance_path': 'Linear.missing'}])
        self.assertEqual(result['components'][0]['status'], 'contradicted')

    def test_top_level_training_loss_can_be_verified(self):
        from types import SimpleNamespace
        torch = self.torch
        loss = lambda logits, labels: ((logits - labels) ** 2).mean()
        candidate = SimpleNamespace(training_loss=loss)
        result = self.run_probe(torch.nn.Linear(2, 1).cuda(),
            [{'id': 'loss', 'instance_path': 'candidate.training_loss'}], candidate)
        self.assertEqual(result['components'][0]['status'], 'verified')
        self.assertIs(candidate.training_loss, loss)

    def test_parameters_created_after_optimizer_construction_are_flagged(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.initial = torch.nn.Linear(2, 1)
            def forward(self, value):
                self.late = torch.nn.Linear(2, 1).cuda()
                return self.initial(value) + self.late(value)
        result = self.run_probe(Model().cuda(), [{'id': 'model', 'instance_path': 'Model'}])
        self.assertFalse(result['optimizer_parameter_set_stable'])

    def test_probe_preserves_forward_and_parameter_updates(self):
        from ctrevo.audit import ComponentProbe
        import copy
        torch = self.torch
        model = torch.nn.Linear(2, 1).cuda()
        control = copy.deepcopy(model)
        probe = ComponentProbe(model, [{'id': 'model', 'instance_path': 'Linear'}])
        observations = []
        for value in (model, control):
            optimizer = torch.optim.SGD(value.parameters(), lr=.01)
            optimizer.zero_grad(set_to_none=True)
            output = value(torch.ones(4, 2, device='cuda'))
            output.sum().backward()
            observations.append((output.detach().clone(), [p.grad.clone() for p in value.parameters()],
                                 torch.cuda.get_rng_state().clone()))
            optimizer.step()
            if value is model:
                probe.after_step()
        probe.close()
        self.assertTrue(torch.equal(observations[0][0], observations[1][0]))
        self.assertTrue(torch.equal(observations[0][2], observations[1][2]))
        for observed, expected in zip(observations[0][1], observations[1][1]):
            self.assertTrue(torch.equal(observed, expected))
        for observed, expected in zip(model.parameters(), control.parameters()):
            self.assertTrue(torch.equal(observed, expected))

    def test_closed_probe_stops_observing_cached_loss_callable(self):
        from ctrevo.audit import ComponentProbe
        from types import SimpleNamespace
        torch = self.torch
        model = torch.nn.Linear(2, 1).cuda()
        candidate = SimpleNamespace(training_loss=lambda output, labels: output.mean())
        probe = ComponentProbe(model, [{'id': 'loss', 'instance_path': 'candidate.training_loss'}],
                               candidate=candidate)
        cached = candidate.training_loss
        probe.close()
        cached(model(torch.ones(4, 2, device='cuda')), None).backward()
        self.assertEqual(probe.result()['components'][0]['targets'][0]['calls'], 0)
