"""Command line entry points; credentials stay in the invoking environment."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from .data import prepare


def main():
    parser = argparse.ArgumentParser(prog='ctrevo')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Prepare all 45,840,617 labeled Criteo rows')
    prep.add_argument('--raw', required=True)
    prep.add_argument('--output', required=True)
    for name in ('search', 'finalize'):
        item = commands.add_parser(name)
        item.add_argument('--data', required=True)
        item.add_argument('--output', required=True)
        item.add_argument('--image', required=True, help='CUDA Docker image (resolved to immutable ID)')
        item.add_argument('--venv', required=True, help='Host Python venv compatible with the image')
        item.add_argument('--seed', type=int, default=42)
        item.add_argument('--epochs', type=int, default=1)
        item.add_argument('--batch-size', type=int, default=8192)
        item.add_argument('--trial-timeout', type=int, default=3600)
        if name == 'search':
            item.add_argument('--provider-url', default='https://api.deepseek.com')
            item.add_argument('--model', default='deepseek-flash')
            item.add_argument('--thinking', choices=['enabled', 'omit'], default='enabled')
            item.add_argument('--steps', type=int, default=2)
            item.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw, args.output)
        return
    if min(args.epochs, args.batch_size, args.trial_timeout) <= 0:
        parser.error('epochs, batch size and timeout must be positive')
    if args.command == 'search' and not os.environ.get('CTR_AGENT_API_KEY'):
        parser.error('set CTR_AGENT_API_KEY in the environment')
    from .task import CTRTask
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], text=True).strip()
    task = CTRTask(args.data, image=image, venv=args.venv, seed=args.seed,
        max_epochs=args.epochs, batch_size=args.batch_size, timeout=args.trial_timeout)
    if args.command == 'finalize':
        print(json.dumps(task.finalize(args.output), indent=2))
        return
    from model_evo_harness import run_search, load_catalog
    from .agent import CTRAgent
    agent = CTRAgent(args.provider_url, os.environ['CTR_AGENT_API_KEY'], args.model,
        thinking=args.thinking, iteration_effort='high', review_effort='max', timeout=600,
        logs=Path(args.output) / 'provider')
    state = run_search(task, agent, output=Path(args.output), catalog=load_catalog(),
                       max_steps=args.steps, resume=args.resume)
    print(json.dumps({'status': state['status'], 'best_id': state['best_id'],
                      'attempts': len(state['steps'])}))


if __name__ == '__main__':
    main()
