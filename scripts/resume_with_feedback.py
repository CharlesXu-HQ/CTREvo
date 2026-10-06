"""Resume an interrupted search with its rejected provider response and exact error.

This changes only Agent context. Evaluation code, dataset and protocol identities
remain subject to the normal Harness resume checks. No candidate code is repaired.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

from ctrevo.agent import CTRAgent
from ctrevo.reply import decode_reply
from ctrevo.task import CTRTask
from model_evo_harness import load_catalog, run_search


class FeedbackAgent(CTRAgent):
    feedback = None

    def propose(self, context):
        if self.feedback:
            context = {**context, **self.feedback}
            self.feedback = None
        return super().propose(context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--error-log', required=True, type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--venv', required=True)
    parser.add_argument('--provider-url', default='https://api.deepseek.com')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--trial-timeout', type=int, default=3600)
    args = parser.parse_args()
    logs = args.output / 'provider'
    last = sorted(logs.glob('call-*.json'))[-1]
    reply = json.loads(last.read_text())['content']
    errors = args.error_log.read_text().splitlines()[-1]
    feedback = {'proposal_error': errors,
        'resume_instruction': 'Repair the rejected response below. Preserve all valid decisions. '
            'candidate must be at the root, horizontal_expansion inside research.model_design. '
            'For retain, copy source mechanism and code_sections verbatim; intentional changes use adapt or retest.'}
    try:
        feedback['rejected_proposal'] = decode_reply(reply)[0]
    except ValueError:
        feedback['invalid_response'] = reply
    # Durable, inspectable provenance for the additional Agent context.
    (args.output / 'resume-feedback.json').write_text(json.dumps({
        'provider_response_file': last.name, 'feedback': feedback}, indent=2))
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], text=True).strip()
    task = CTRTask(args.data, image=image, venv=args.venv, seed=args.seed, max_epochs=args.epochs,
                   batch_size=args.batch_size, timeout=args.trial_timeout)
    agent = FeedbackAgent(args.provider_url, os.environ['CTR_AGENT_API_KEY'], args.model,
        thinking='enabled', iteration_effort='high', review_effort='max', timeout=600, logs=logs)
    agent.feedback = feedback
    state = run_search(task, agent, output=args.output, catalog=load_catalog(), max_steps=args.steps, resume=True)
    print(json.dumps({'status': state['status'], 'best_id': state['best_id']}))


if __name__ == '__main__':
    main()
