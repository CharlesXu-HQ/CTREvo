"""Replay saved candidates to verify the host contract; no new Agent search."""
import argparse
import json
import subprocess
from pathlib import Path

from ctrevo.task import CTRTask
from model_evo_harness import call_with_references, load_catalog, read_references, run_search

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', required=True)
parser.add_argument('--image', required=True)
parser.add_argument('--venv', required=True)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--journal', type=Path, default=Path(__file__).parents[1] / 'criteo-full-20261006/journal.json')
args = parser.parse_args()
saved = json.loads(args.journal.read_text())
catalog = load_catalog()


class Replay:
    def propose(self, context):
        proposal = saved['steps'][len(context['steps'])]['proposal']
        material = read_references(catalog, {'framework': 'pytorch', 'include_training': True})
        return call_with_references(lambda supplied: proposal,
                                   {**context, 'reference_material': material['files']})

    def reflect(self, observation):
        trial = observation['trial']
        measured = trial.get('evaluation', {})
        evidence = json.dumps({'metrics': measured.get('metrics'),
                               'implementation': measured.get('implementation_check', {}).get('status')})
        return {'technical_experience': {
            'lesson': 'Saved candidate replay verifies execution and selection plumbing only.',
            'evidence': evidence,
            'uncertainty': 'No fresh Agent decisions or component-benefit attribution in this replay.',
            'next_test': 'Use a separately budgeted live Agent search for new hypotheses.',
            'attribution': 'unverified',
            'component_assessments': [{'component_id': component['id'], 'outcome': 'inconclusive',
                'evidence': 'Host component probe and independent whole-recipe metrics are available.',
                'compatibility_limits': 'Bounded training batches; no sharing or mathematical-equivalence proof.',
                'next_test': 'A controlled ablation is needed to attribute benefit.', 'attribution': 'unverified'}
                for component in trial['proposal']['research']['model_design']['components']]},
            'business_experience': {'status': 'not_observable', 'reason': 'Anonymous CTR fields.'}}


image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                 args.image], text=True).strip()
task = CTRTask(args.data, image=image, venv=args.venv)
state = run_search(task, Replay(), output=args.output, catalog=catalog,
                   max_steps=2, require_verified_implementation=True)
report = {'kind': 'saved_candidate_replay_not_new_agent_search', 'status': state['status'],
    'dataset_digest': task.manifest['dataset_digest'], 'split_rows': task.manifest['split_rows'],
    'identity': state['identity'], 'best_id': state['best_id'],
    'baseline': {'metrics': state['baseline']['metrics'],
                 'implementation_check': state['baseline']['implementation_check']},
    'trials': [{'id': step['id'], 'status': step['status'], 'promotion': step['promotion'],
               'metrics': step.get('evaluation', {}).get('metrics'),
               'implementation_check': step.get('evaluation', {}).get('implementation_check'),
               'change_audit': step.get('evaluation', {}).get('change_audit'),
               'rows_seen': step.get('evaluation', {}).get('runtime', {}).get('rows_seen'),
               'reference_events': step['proposal'].get('reference_events')}
              for step in state['steps']]}
(args.output / 'verification-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'best_id': state['best_id'], 'baseline_check': state['baseline']['implementation_check']['status'],
                  'trials': [(item['id'], item['status'], item['promotion']) for item in state['steps']]}))
