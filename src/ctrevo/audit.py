"""Host observations of declared component execution, separate from attribution."""

import hashlib
import math
from functools import wraps


def audit_execution(source, components, runtime, *, train_rows, prediction_rows):
    probe = runtime.get('implementation_probe', {})
    findings = []
    status = 'verified'
    if (runtime.get('device') != 'cuda' or runtime.get('rows_seen') != train_rows
            or runtime.get('prediction_rows') != prediction_rows):
        findings.append('Full-data CUDA execution contract is not satisfied')
        status = 'contradicted'
    if probe.get('optimizer_parameter_set_stable') is False:
        findings.append('Model parameters changed after optimizer construction')
        status = 'contradicted'
    elif probe.get('optimizer_parameter_set_stable') is not True and status != 'contradicted':
        findings.append('Optimizer parameter coverage was not observed')
        status = 'unverified'
    observed = {item['id']: item for item in probe.get('components', [])}
    if not components or not probe.get('batches'):
        findings.append('No bounded component execution probe is available')
        if status != 'contradicted':
            status = 'unverified'
    digest = hashlib.sha256(source.encode()).hexdigest()
    if probe.get('source_sha256') not in (None, digest):
        findings.append('Probe source differs from the evaluated candidate')
        status = 'contradicted'
    elif probe.get('source_sha256') is None and status != 'contradicted':
        status = 'unverified'
    for component in components:
        item = observed.get(component['id'])
        if item is None:
            findings.append(f"No observation for component {component['id']}")
            if status != 'contradicted':
                status = 'unverified'
        elif item.get('instance_path') != component.get('instance_path'):
            findings.append(f"Binding differs for component {component['id']}")
            status = 'contradicted'
        elif item.get('status') == 'contradicted':
            findings.append(f"Declared path not found for component {component['id']}")
            status = 'contradicted'
        elif item.get('status') != 'verified':
            findings.append(f"Execution remains unverified for component {component['id']}")
            if status != 'contradicted':
                status = 'unverified'
    return {'status': status, 'scope': 'declared_component_execution_v1',
            'findings': findings, 'components': probe.get('components', []),
            'probe_batches': probe.get('batches', 0),
            'limitations': 'Bounded execution/gradient observations do not prove mathematical '
                          'equivalence to a named method, declared parameter sharing, exact fusion '
                          'topology, all input paths, or gain attribution.'}


class ComponentProbe:
    """Observe real training batches without extra forwards or optimizer updates.

    Components may name modules, ModuleList/ModuleDict containers, or bound
    methods. Unobserved or disconnected paths remain unverified. This is a
    diagnostic for cooperative candidate code, not isolation from malicious code.
    """

    def __init__(self, model, components, *, candidate=None):
        from torch import nn
        self.model = model
        self.components = []
        self.handles = []
        self.restorations = []
        self.batches = 0
        self.connected_parameters = set()
        self.active = True
        self.parameter_names = {id(p): name for name, p in model.named_parameters()}
        for component in components:
            record = {'id': component['id'], 'instance_path': component.get('instance_path'),
                      'targets': []}
            self.components.append(record)
            path = component.get('instance_path')
            if not isinstance(path, str) or not path:
                record['missing'] = True
                continue
            parts = path.split('.')
            parent, target = None, model
            if parts[0] == 'candidate':
                target, parts = candidate, parts[1:]
            elif parts[0] == 'training_loss':
                target = candidate
            elif parts[0] in (type(model).__name__, 'model'):
                parts = parts[1:]
            try:
                for part in parts:
                    parent, target = target, getattr(target, part)
            except AttributeError:
                record['missing'] = True
                continue
            if isinstance(target, (nn.ModuleList, nn.ModuleDict)):
                for name, child in target.named_children():
                    self._module(child, record, f'{path}.{name}')
            elif isinstance(target, nn.Module):
                self._module(target, record, path)
            elif callable(target) and parent is not None:
                observation = self._target(record, path, [])
                original_local = vars(parent).get(parts[-1])
                had_local = parts[-1] in vars(parent)
                setattr(parent, parts[-1], self._method(target, observation))
                self.restorations.append((parent, parts[-1], had_local, original_local))
            else:
                record['missing'] = True

    def _target(self, record, path, parameters):
        result = {'path': path, 'calls': 0, 'nonzero_output_gradient': False,
                  'trainable_parameters': parameters, 'output_shapes': [],
                  'output_summaries': []}
        record['targets'].append(result)
        return result

    def _module(self, module, record, path):
        parameters = [self.parameter_names[id(p)] for p in module.parameters() if p.requires_grad]
        observation = self._target(record, path, parameters)
        self.handles.append(module.register_forward_hook(
            lambda _module, _inputs, output: self._output(output, observation)))

    def _method(self, method, observation):
        @wraps(method)
        def call(*args, **kwargs):
            output = method(*args, **kwargs)
            if self.active:
                self._output(output, observation)
            return output
        return call

    def _output(self, output, observation):
        import torch
        observation['calls'] += 1

        def capture(value):
            if isinstance(value, torch.Tensor):
                shape = list(value.shape)
                if shape not in observation['output_shapes']:
                    observation['output_shapes'].append(shape)
                if len(observation['output_summaries']) < 3:
                    with torch.no_grad():
                        sample = value.detach().double()
                        finite = bool(torch.isfinite(sample).all())
                        statistics = {'mean': None, 'std': None, 'max_abs': None}
                        if finite and sample.numel():
                            statistics = {'mean': sample.mean().item(),
                                          'std': sample.std(unbiased=False).item(),
                                          'max_abs': sample.abs().max().item()}
                            if not all(math.isfinite(number) for number in statistics.values()):
                                finite = False
                                statistics = {key: None for key in statistics}
                        observation['output_summaries'].append(
                            {'shape': shape, **statistics, 'finite': finite})
                if value.requires_grad:
                    def backward(gradient):
                        if bool(torch.isfinite(gradient).all()) and bool(torch.any(gradient != 0)):
                            observation['nonzero_output_gradient'] = True
                    self.handles.append(value.register_hook(backward))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    capture(item)
            elif isinstance(value, dict):
                for item in value.values():
                    capture(item)
        capture(output)

    def after_step(self):
        import torch
        self.batches += 1
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()):
                self.connected_parameters.add(name)

    def result(self):
        for component in self.components:
            for target in component['targets']:
                target['parameters_with_gradient'] = [name for name in target['trainable_parameters']
                                                      if name in self.connected_parameters]
                target['status'] = 'verified' if (target['calls'] and target['nonzero_output_gradient']
                    and len(target['parameters_with_gradient']) == len(target['trainable_parameters'])) else 'unverified'
            component['status'] = ('contradicted' if component.get('missing') else
                'verified' if component['targets'] and all(t['status'] == 'verified'
                    for t in component['targets']) else 'unverified')
        return {'batches': self.batches, 'components': self.components,
                'optimizer_parameter_set_stable': set(self.parameter_names) ==
                    {id(p) for p in self.model.parameters()}}

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for parent, name, had_local, original in reversed(self.restorations):
            if had_local:
                setattr(parent, name, original)
            else:
                delattr(parent, name)
        self.restorations.clear()
