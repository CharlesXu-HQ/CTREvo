"""CTR-specific provider adapter using the shared Harness contracts."""

import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from model_evo_harness import (COMPOSITION_INSTRUCTIONS, REFERENCE_INSTRUCTIONS,
    propose_with_references, validate_research, validate_reflection, validate_data_request)
from model_evo_harness.provider import OpenAICompatibleAgent, _PROPOSE_INSTRUCTIONS

from .execution import validate_candidate
from .reply import decode_reply


CONTRACT = """You lead a CTR research task using the supplied ModelEvoHarness. Return JSON only.
Choose experiment, stop(reason), or request_data(request). Experiment contains candidate:{source,config}
and research. source is COMPLETE Python source, exports build_model(schema,config) -> native torch.nn.Module.
Module.forward(dense,categorical) returns finite CUDA logits of shape [batch]. Never sigmoid the output.
schema: dense_width=26 (13 standardized original numeric inputs followed by missing indicators),
categorical_width=26, buckets=65536 unless task says otherwise. categorical is int64 [batch,26],
each field in [0,buckets). Use field-specific embeddings or offsets. build_model config is config.model.
Optional training_loss(logits,labels) returns a scalar. Host owns AdamW, gradient clipping at 10,
full epoch coverage, fixed batch size, GPU, evaluator and held-out labels. Candidate config keys:
lr (1e-6..0.1), weight_decay (0..0.1), epochs (1..task.max_epochs), model (arbitrary JSON object).
Defaults are lr=.002, weight_decay=1e-6, epochs=1, model={}. Do not add batch_size or optimizer keys.
Imports: torch, numpy, math, typing, collections and exact model_evo_harness.models.pytorch modules.
Use baseline_candidate and previous step candidate sources as editable code. Be explicit about
trainable modules, forward paths, fusion and partial sharing. You can change real structure and loss,
not merely select a named model. Prefer one informative change with a viable budget control.
Research requires nonempty direction, mechanism, why_now, data_rationale, comparison, expected_result,
falsification; input_fields uses ORIGINAL I1..I13/C1..C26 names; alternatives is nonempty list of
{direction,mechanism,reason}; cite observed evidence_ids and list change_factors. Include model_design
and horizontal_expansion per the composition instructions. Exact formatting matters:
EVERY component.output_contract must be a STRING, e.g. "float32 [B,16], unbounded, no mask".
For EVERY branch and fusion listed in horizontal_expansion.groups, component.code_sections
must include an ACTUAL forward/call entry point, e.g. ["CTRModel.forward", "CTRModel.fm_logit"].
A helper name such as CTRModel.fm_logit alone is insufficient. Also provide instance_path
as a nonempty string. Keep these invariants in every repair; do not convert strings to objects.
instance_path must resolve on the built model: e.g. CTRModel.embedding, CTRModel.cross_layers,
or CTRModel.fm_logit for a bound method; use candidate.training_loss for a custom loss component.
Register all trainable parameters in __init__, before the host constructs its optimizer.
Host probes the first three real training batches:
each declared component must execute and its tensor output receive finite nonzero loss gradient;
trainable module parameters must receive finite gradients. ModuleList/ModuleDict children are
checked individually. Unobserved paths remain unverified and cannot be promoted in strict mode.
This verifies bounded component execution, NOT named-model mathematical equivalence, declared
parameter sharing, exact fusion topology, or gain attribution.
Return exactly one JSON object without trailing prose or a second JSON object.
When composition_sources is empty: change_scope="initialize", parent_trial_id=null,
inheritance=[]. The untracked baseline is a comparison, NOT a registered inheritance source.
Before drafting code, use read_reference for intended bundled methods and include_composition=true;
this avoids generating a draft that must be discarded to fetch source. Novel modules are still allowed.
First tracked design is initialize;
later local changes name a real prior trial and account for EVERY parent component.
Anonymous inputs have no known user/item/sequence semantics. Do not invent business segments,
timestamps or a reason to apply DIN. Numeric/categorical groups are typed views, not business entities.
Minimize validation LogLoss. AUC, calibration and resource use are secondary. Intervals are exploratory.
If proposal_error or reflection_error is supplied, repair the rejected response specifically.
No final holdout metrics are available. New fields require the Harness's justified data-request rules.
"""

REFLECT = """Analyze this executed CTR trial against its hypothesis, control and actual runtime.
Return technical_experience:{lesson,evidence,uncertainty,next_test,attribution:'unverified',
component_assessments:[{component_id,outcome:promising|inconclusive|harmful|invalid,evidence,
compatibility_limits,next_test,attribution:'unverified'}]} plus
business_experience:{status:'not_observable',reason:'Anonymous CTR features have no observable business semantics'}.
Assess every current model_design component exactly once, including branches and fusion.
Use actual metrics/code/runtime. CUDA execution and gradient presence do not prove a proposed mechanism
caused improvement. Joint changes, shared-weight retraining and adaptive selection limit attribution.
implementation_check is only a bounded execution check. Even when verified, change_audit and
component benefit attribution remain unverified until an appropriate comparison is executed.
Failed trials cannot establish promising/harmful component outcomes. Explain the next discriminating
experiment within current data and budget. Preserve uncertainty and dataset-specific applicability.
"""


class CTRAgent(OpenAICompatibleAgent):
    def __init__(self, *args, logs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = Path(logs) if logs else None
        self.calls = 0

    def _complete(self, instructions, context, effort):
        self.calls += 1
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': instructions},
                    {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}],
                   'response_format': {'type': 'json_object'}, 'reasoning_effort': effort,
                   'max_tokens': 32768}
        if self.thinking == 'enabled':
            payload['thinking'] = {'type': 'enabled'}
        request = Request(self.url, data=json.dumps(payload).encode(), method='POST',
                          headers={'Authorization': f'Bearer {self._api_key}', 'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                envelope = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f'provider HTTP {error.code}') from None
        except Exception:
            raise RuntimeError('provider request failed') from None
        choice = envelope.get('choices', [{}])[0]
        content = choice.get('message', {}).get('content', '')
        if self.logs:
            self.logs.mkdir(parents=True, exist_ok=True)
            (self.logs / f'call-{time.time_ns()}-{self.calls:03d}.json').write_text(json.dumps({
                'effort': effort, 'seconds': time.monotonic() - started,
                'finish_reason': choice.get('finish_reason'), 'usage': envelope.get('usage'),
                'content': content}, indent=2))
        if choice.get('finish_reason') != 'stop':
            raise ValueError('provider response incomplete; return concise complete JSON')
        try:
            value, repaired = decode_reply(content)
            if repaired and self.logs:
                with (self.logs / 'framing-repairs.jsonl').open('a') as stream:
                    stream.write(json.dumps({'call': self.calls, 'terminal_delimiters_only': True}) + '\n')
        except (ValueError, TypeError) as error:
            failure = ValueError(f'invalid final JSON: {error}; repair syntax without changing the proposal')
            failure.raw_response = content
            raise failure from None
        if not isinstance(value, dict):
            raise ValueError('response must be a JSON object')
        return value

    def propose(self, context):
        current = dict(context)
        ledger = {}
        errors = []
        for attempt in range(3):
            answer = None
            try:
                answer = propose_with_references(
                    lambda c: self._complete(_PROPOSE_INSTRUCTIONS + CONTRACT + COMPOSITION_INSTRUCTIONS + REFERENCE_INSTRUCTIONS,
                                             c, self.iteration_effort),
                    current, catalog=context['catalog'], framework='pytorch', read_state=ledger)
                if answer.get('action') == 'experiment':
                    validate_candidate(answer.get('candidate'), max_epochs=context['task']['max_epochs'])
                    answer['research'] = validate_research(answer, context['task'], context['catalog'],
                        evidence=context.get('evidence'), sources=context.get('composition_sources', []))
                elif answer.get('action') == 'request_data':
                    validate_data_request(answer.get('request'), context['task'], context['steps'],
                                          evidence=context.get('evidence'))
                elif answer.get('action') != 'stop' or not answer.get('reason'):
                    raise ValueError('need experiment, justified request_data, or stop with reason')
                return answer
            except (ValueError, SyntaxError) as error:
                if attempt == 2:
                    raise
                if hasattr(error, 'raw_response'):
                    current['invalid_response'] = error.raw_response
                errors.append(str(error)[:2000])
                current.update(proposal_error=errors[-1], proposal_errors=errors.copy())
                if answer is not None:
                    current['rejected_proposal'] = answer
                if self.logs:
                    self.logs.mkdir(parents=True, exist_ok=True)
                    with (self.logs / 'validation-errors.jsonl').open('a') as stream:
                        stream.write(json.dumps({'attempt': attempt + 1, 'error': errors[-1]}) + '\n')

    def reflect(self, observation):
        current = dict(observation)
        trial = observation['trial']
        evaluation = {**trial.get('evaluation', {}), 'research': trial['proposal']['research'],
                      'trial_status': trial['status']}
        if self.logs:
            log = self.logs.parent / trial['id'] / 'worker.log'
            if trial['status'] == 'failed' and log.is_file():
                current['execution_diagnostic'] = log.read_text(errors='replace')[-6000:]
        effort = self.review_effort if evaluation.get('review_required') or trial['status'] == 'failed' else self.iteration_effort
        for attempt in range(3):
            answer = None
            try:
                answer = self._complete(REFLECT + COMPOSITION_INSTRUCTIONS, current, effort)
                return validate_reflection(answer, evaluation, observation['task'],
                    [*observation['steps'], trial], evidence=observation.get('evidence'))
            except ValueError as error:
                if attempt == 2:
                    raise
                current.update(reflection_error=str(error)[:2000], rejected_reflection=answer)
