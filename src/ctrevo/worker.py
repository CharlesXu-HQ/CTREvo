"""Trusted GPU training protocol. Evaluation labels are never mounted here."""

import importlib.util
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit import ComponentProbe


def batches(folder, size, *, rng=None, labels=False):
    dense = np.load(folder / 'dense.npy', mmap_mode='r')
    cat = np.load(folder / 'categorical.npy', mmap_mode='r')
    target = np.load(folder / 'labels.npy', mmap_mode='r') if labels else None
    starts = np.arange(0, len(dense), size)
    if rng is not None:
        rng.shuffle(starts)
    for start in starts:
        end = min(start + size, len(dense))
        x = torch.tensor(np.asarray(dense[start:end]), dtype=torch.float32, device='cuda')
        c = torch.tensor(np.asarray(cat[start:end], dtype=np.int64), device='cuda')
        y = torch.tensor(target[start:end], dtype=torch.float32, device='cuda') if labels else None
        yield start, end, x, c, y


def logits(model, x, c):
    result = model(x, c)
    if not isinstance(result, torch.Tensor) or result.shape != (len(x),) or result.device.type != 'cuda':
        raise ValueError('model must produce [batch] CUDA logits')
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite logits')
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    torch.set_num_threads(4)
    job = json.loads(Path('/input/job.json').read_text())
    torch.manual_seed(job['seed'])
    torch.cuda.manual_seed_all(job['seed'])
    np.random.seed(job['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = importlib.util.spec_from_file_location('candidate', '/input/candidate.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    model = candidate.build_model(job['schema'], job['config']['model']).cuda()
    if not isinstance(model, nn.Module):
        raise ValueError('build_model must return a torch module')
    start = time.monotonic()
    rows_seen, steps, last_loss = 0, 0, None
    gradient_norms = {}
    probe = None if job['checkpoint'] else ComponentProbe(model, job.get('components', []), candidate=candidate)
    if job['checkpoint']:
        model.load_state_dict(torch.load('/checkpoint/weights.pt', map_location='cuda', weights_only=True))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=job['config']['lr'],
                                      weight_decay=job['config']['weight_decay'])
        loss_fn = getattr(candidate, 'training_loss', nn.functional.binary_cross_entropy_with_logits)
        for epoch in range(job['config']['epochs']):
            model.train()
            for _, _, x, c, y in batches(Path('/train'), job['batch_size'],
                    rng=np.random.default_rng(job['seed'] + epoch), labels=True):
                optimizer.zero_grad(set_to_none=True)
                output = logits(model, x, c)
                loss = loss_fn(output, y)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError('training_loss must return a finite scalar')
                loss.backward()
                if not steps:
                    gradient_norms = {name: None if p.grad is None else float(p.grad.norm())
                                      for name, p in model.named_parameters()}
                    if not any(v is not None and v > 0 for v in gradient_norms.values()):
                        raise ValueError('model has no nonzero gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                if steps < 3:
                    probe.after_step()
                    if steps == 2:
                        probe.close()
                rows_seen += len(x)
                steps += 1
                last_loss = float(loss.detach())
                if steps % 250 == 0:
                    print(json.dumps({'epoch': epoch + 1, 'rows_seen': rows_seen,
                                      'loss': last_loss, 'seconds': time.monotonic() - start}), flush=True)
        torch.save(model.state_dict(), '/output/weights.pt')
        probe.close()
    model.eval()
    rows = len(np.load('/target/dense.npy', mmap_mode='r'))
    prediction = np.lib.format.open_memmap('/output/prediction.npy', mode='w+', dtype='float32', shape=(rows,))
    with torch.no_grad():
        for lo, hi, x, c, _ in batches(Path('/target'), job['batch_size']):
            prediction[lo:hi] = logits(model, x, c).sigmoid().cpu().numpy()
    prediction.flush()
    runtime = {'device': 'cuda', 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'rows_seen': rows_seen, 'steps': steps, 'prediction_rows': rows, 'last_loss': last_loss,
        'seconds': time.monotonic() - start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'parameters': sum(p.numel() for p in model.parameters()), 'config': job['config'],
        'first_batch_gradient_norms': gradient_norms,
        'implementation_probe': ({**probe.result(),
            'source_sha256': hashlib.sha256(Path('/input/candidate.py').read_bytes()).hexdigest(),
            'training_loss': getattr(loss_fn, '__name__', type(loss_fn).__name__)} if probe else {}),
        'modules': [{'path': name, 'class': type(module).__name__} for name, module in model.named_modules()]}
    Path('/output/runtime.json').write_text(json.dumps(runtime, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
