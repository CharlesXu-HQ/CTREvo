"""Fixed-plan full-data CUDA verification; not a new DeepSeek optimization run."""
import argparse
import json
from pathlib import Path
import subprocess

from ctrevo.task import CTRTask
from model_evo_harness import call_with_references, load_catalog, read_references, run_search


class InteractionVerification:
    def __init__(self, source, catalog):
        self.source, self.catalog = source, catalog

    def propose(self, context):
        index = len(context['steps'])
        task = context['task']
        fields = task['fields']
        selected = 'mixed' if index == 0 else 'numeric'
        enabled = ['cc', 'nc'] if index == 0 else ['cc', 'nc', 'nn']
        control = ['cc'] if index == 0 else ['cc', 'nc']
        comparison = 'Same source, seed, optimizer and full-data protocol; gate the selected term at fusion.'
        components = []
        for name, scope, mechanism, output in [
            ('embedding', fields[13:], 'Shared categorical field embedding table', 'float32 [B,26,16]'),
            ('network', fields, 'Original embedding MLP including numeric values and missing indicators', 'float32 [B,1]'),
            ('numeric_embedding', fields[:13], 'Observed standardized x_i times a learned per-field vector', 'float32 [B,13,16]'),
            ('grouped_fm', fields, 'Three second-order group sums cc, nc, nn over field vectors', 'float32 [B,3]'),
            ('fusion', fields, 'Deep logit plus .05 times configured cc/nc/nn terms', 'float32 [B]'),
        ]:
            components.append({'id': name, 'mechanism': mechanism, 'input_fields': scope,
                'required_capabilities': ['tabular_features'], 'instance_path': f'CTRModel.{name}',
                'code_sections': ['CTRModel.__init__', 'CTRModel.forward'],
                'output_contract': output + ', unbounded'})
        design = {'estimator': 'Pointwise binary logit with host BCE', 'estimator_id': 'pointwise_bce',
            'backbone': 'Embedding MLP with grouped second-order residuals', 'backbone_id': 'embedding_mlp',
            'change_scope': 'initialize' if index == 0 else 'local',
            'parent_trial_id': None if index == 0 else context['steps'][-1]['id'],
            'rationale': 'Bounded integration verification using an editable mixed-field example',
            'data_fit': 'Existing typed anonymous inputs only; numeric missing values masked before FM',
            'comparison_plan': comparison, 'components': components, 'inheritance': [],
            'horizontal_expansion': {'decision': 'expand' if index == 0 else 'defer',
                'rationale': 'Parallel MLP and explicit pair terms; later trial changes term gating only',
                'comparison_plan': comparison,
                'groups': [{'id': 'mixed_interactions', 'branch_ids': ['network', 'grouped_fm'],
                    'fusion_id': 'fusion', 'parameter_sharing': [{
                        'component_ids': ['network', 'grouped_fm'], 'code_sections': ['CTRModel.embedding'],
                        'rationale': 'Same field embedding vectors feed the MLP and grouped pair branch'}]}]}}
        if index:
            design['inheritance'] = [{'source_trial_id': context['steps'][-1]['id'],
                'component_id': item['id'], 'target_component_id': item['id'], 'decision': 'retest',
                'reason': 'Repeat under a changed fusion gate; no assumed benefit',
                'compatibility': 'Same field schema, masks and backbone',
                'validation_plan': comparison} for item in components]
        coverage = []
        for pair in context['interaction_context']['coverage_pairs']:
            has_explicit = index > 0 and 'missing' not in pair.values() and pair != {'left': 'numeric', 'right': 'numeric'}
            coverage.append({**pair, 'status': 'mixed' if has_explicit else 'implicit',
                'basis': 'Current MLP consumes all views; previous example explicitly enables cc/nc only' if index
                         else 'Seed concatenates standardized values, missing flags and embeddings into its MLP'})
        research = {'direction': 'Explicit interaction scope', 'mechanism': f'Enable {selected} terms at fixed scale',
            'why_now': 'Exercise available typed inputs and the new controlled-training contract; no claim of optimal priority',
            'data_rationale': '39 anonymous original fields, no invented business or sequence semantics',
            'comparison': comparison, 'expected_result': 'Executable full-data comparison; lower candidate logloss would favor this recipe',
            'falsification': 'Nonnegative paired difference rejects a useful gain under this protocol',
            'input_fields': fields, 'evidence_ids': ['dataset.train', 'baseline.metrics'],
            'change_factors': ['enabled interaction scope'],
            'alternatives': [{'direction': 'fusion diagnosis', 'mechanism': 'inspect scales before increasing capacity',
                              'reason': 'Bounded probe supplies scale observations; defer extra structure'}],
            'model_design': design,
            'interaction_plan': {'decision': 'test', 'rationale': 'Fixed integration plan, not an Agent priority judgment',
                'coverage': coverage, 'candidates': [{'id': selected,
                    'views': ['numeric', 'categorical'] if index == 0 else ['numeric'],
                    'order': 2, 'mechanism': 'grouped field-vector dot products', 'priority': 1,
                    'reason': 'Host declares these available typed fields; test scope with an actual control',
                    'cost': 'One full training and one full control',
                    'risks': 'Shared-gradient effects, scale and anonymous sparse-field support',
                    'evidence_ids': ['dataset.train', 'baseline.metrics']}], 'selected_id': selected,
                'control': {'config_patch': {'model': {'enabled_pairs': control}},
                            'expected_effect': 'Remove only the newly enabled term from the residual sum'}}}
        proposal = {'action': 'experiment', 'candidate': {'source': self.source,
                    'config': {'model': {'enabled_pairs': enabled, 'fm_scale': .05}}}, 'research': research}
        material = read_references(self.catalog, {'framework': 'pytorch', 'include_interactions': True})
        return call_with_references(lambda supplied: proposal,
                                   {**context, 'reference_material': material['files']})

    def reflect(self, observation):
        trial = observation['trial']
        evaluation = trial.get('evaluation', {})
        return {'technical_experience': {'lesson': 'Fixed-plan contract verification with measured recipe comparisons',
            'evidence': json.dumps({'score': evaluation.get('score'),
                'control': evaluation.get('interaction_control'), 'status': trial['status']}),
            'uncertainty': 'No new LLM decisions, multiple seeds, isolated component attribution or final holdout',
            'next_test': 'Let the Agent compare observed scope outcomes and branch scales before new structure',
            'attribution': 'unverified', 'component_assessments': [
                {'component_id': item['id'], 'outcome': 'inconclusive',
                 'evidence': 'Bounded probe and controlled whole-recipe results; no individual field-pair effect estimate',
                 'compatibility_limits': 'Current anonymous Criteo views and fixed training protocol',
                 'next_test': 'Use the executed control and diagnosis to design the next comparison',
                 'attribution': 'unverified'} for item in trial['proposal']['research']['model_design']['components']]},
            'business_experience': {'status': 'not_observable', 'reason': 'Anonymous field semantics'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('data', 'image', 'venv', 'output'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], text=True).strip()
    task = CTRTask(args.data, image=image, venv=args.venv)
    catalog = load_catalog()
    agent = InteractionVerification((root / 'examples/mixed_fm.py').read_text(), catalog)
    state = run_search(task, agent, output=Path(args.output), catalog=catalog,
                       max_steps=2, require_verified_implementation=True)
    report = {'kind': 'fixed_plan_contract_verification_not_new_agent_search',
        'status': state['status'], 'best_id': state['best_id'], 'identity': state['identity'],
        'dataset_digest': task.manifest['dataset_digest'], 'split_rows': task.manifest['split_rows'],
        'baseline': state['baseline'], 'steps': state['steps']}
    (Path(args.output) / 'verification-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': state['status'], 'best': state['best_id'],
        'trials': [(s['id'], s['status'], s.get('evaluation', {}).get('score'),
                    s.get('evaluation', {}).get('interaction_control', {}).get('status')) for s in state['steps']]}))


if __name__ == '__main__':
    main()
