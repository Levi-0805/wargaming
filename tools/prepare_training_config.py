"""Validate UE's generated settings and bind a published target policy to a batch.

The UI owns scenario/seat/bot selection. This helper cannot prove those choices;
it validates the Python-visible settings and writes a separate effective config.
It never edits the platform-generated input or silently repairs mismatches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import commentjson

ROOT = Path(__file__).resolve().parents[1]
TEAM = 'FsLutk7wD2RkNgbs'
RED_MODULE = f'TeamAlg.{TEAM}.Algorithm.RLAgentAlgorithm'
BLUE_MODULE = 'TeamAlg.BlueModel.Algorithm.RLAgentAlgorithm'
POLICY_KEY = RED_MODULE + '.py->AlgorithmConfig'
DEFAULT_TEMPLATE = ROOT / 'PythonCode/TaskConfig/cssim/iql_eval_red.jsonc'
FLAGS = ('enable_parent_assignment', 'enable_preservation_pause',
         'enable_shared_contact_maneuver', 'enable_forced_entry_under_contact',
         'enable_continuous_column', 'enable_entry_waves')
POLICY_FIELDS = ('mode', 'iql_model', 'iql_shadow', *FLAGS)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize_pointer(value):
    return value.replace('.py->', '->') if isinstance(value, str) else value


def read_config(path):
    raw = path.read_bytes()
    parsed = commentjson.loads(raw.decode('utf-8-sig'))
    if not isinstance(parsed, dict):
        raise ValueError(f'Configuration must be a JSON object: {path}')
    return raw, parsed


def section(config, name):
    found = [(key, value) for key, value in config.items()
             if key.split('->')[-1] == name]
    if len(found) != 1 or not isinstance(found[0][1], dict):
        raise ValueError(f'Expected exactly one object section for {name}')
    return found[0][1]


def require(settings, key, expected, section_name):
    actual = settings.get(key)
    # JSON booleans must not be accepted as numbers (True == 1 in Python).
    if isinstance(expected, bool):
        valid = type(actual) is bool and actual == expected
    elif isinstance(expected, int):
        valid = type(actual) is int and actual == expected
    elif isinstance(expected, float):
        valid = type(actual) in (int, float) and actual == expected
    else:
        valid = type(actual) is type(expected) and actual == expected
    if not valid:
        raise ValueError(f'{section_name}.{key}: expected {expected!r}, got {actual!r}; correct the platform settings')


def validate_platform(config):
    global_config = section(config, 'GlobalConfig')
    scenario = section(config, 'ScenarioConfig')
    for key, expected in dict(num_worker_threads=1, device='cpu', max_episode=10, seed=10098).items():
        require(global_config, key, expected, 'GlobalConfig')
    for key, expected in dict(MaxStepsPerEpisode=500, StepGameTime=.5, TimeDilation=1.,
                              render=True, Enable_Reward_Sharing=False).items():
        require(scenario, key, expected, 'ScenarioConfig')
    for key, module in (('AlgorithmConfig_Red', RED_MODULE), ('AlgorithmConfig_Blue', BLUE_MODULE)):
        if normalize_pointer(global_config.get(key)) != module + '->AlgorithmConfig':
            raise ValueError(f'GlobalConfig.{key} must select {module}')
    names = scenario.get('Algorithm_Names')
    expected = [m + '->ReinforceAgentAlgorithm' for m in (RED_MODULE, BLUE_MODULE)]
    if not isinstance(names, list) or [normalize_pointer(n) for n in names] != expected:
        raise ValueError('ScenarioConfig.Algorithm_Names must select the requested red team and BlueModel in order')


def policy_section(config, required):
    found = [(key, value) for key, value in config.items()
             if normalize_pointer(key) == normalize_pointer(POLICY_KEY)]
    if len(found) > 1 or (found and not isinstance(found[0][1], dict)):
        raise ValueError('Ambiguous or malformed red policy override section')
    if not found:
        if required:
            raise ValueError('Published template has no red policy override')
        return POLICY_KEY, {}
    return found[0]


def prepare(source, output, *, template=DEFAULT_TEMPLATE, cssim_root=Path('C:/CSSIM'), pythoncode_root=None):
    source, output, template = (Path(p).resolve() for p in (source, output, template))
    provenance_path = output.with_name(output.stem + '.provenance.json')
    if output in (source, template) or provenance_path in (source, template):
        raise ValueError('Effective config and provenance must not overwrite an input')
    if output.exists() or provenance_path.exists():
        raise ValueError('Batch output already exists; use a new batch directory')
    source_bytes, original = read_config(source)
    template_bytes, published = read_config(template)
    validate_platform(original)
    _, policy = policy_section(published, required=True)
    if policy.get('mode') != 'hybrid_iql' or policy.get('iql_shadow') is not False:
        raise ValueError('Published template must enable hybrid_iql with iql_shadow=false')
    unknown = set(policy) - set(POLICY_FIELDS)
    if unknown:
        raise ValueError(f'Unexpected policy fields in template: {sorted(unknown)}')
    for flag in FLAGS:
        if flag in policy and type(policy[flag]) is not bool:
            raise ValueError(f'Template {flag} must be a JSON boolean')
    configured_model = policy.get('iql_model')
    if not isinstance(configured_model, str) or not configured_model:
        raise ValueError('Published template must name an IQL model')
    code_root = (Path(pythoncode_root) if pythoncode_root is not None else
                 Path(cssim_root) / 'Client/Data/AlgData/PythonCode').resolve()
    team_root = code_root / 'TeamAlg' / TEAM
    model = Path(configured_model)
    if not model.is_absolute():
        model = team_root / model
    model = model.resolve()
    if not model.is_relative_to(team_root.resolve()):
        raise ValueError('Published model must be inside the selected team directory')
    if not model.is_file():
        raise ValueError(f'IQL model does not exist: {model}')
    injected = copy.deepcopy(policy)
    injected['iql_model'] = str(model)
    effective = copy.deepcopy(original)
    key, existing = policy_section(effective, required=False)
    updated = copy.deepcopy(existing)
    changes = []
    for field, value in injected.items():
        if field not in existing or existing[field] != value:
            changes.append(dict(path=[key, field], before_present=field in existing,
                                before=existing.get(field), after=value))
        updated[field] = value
    effective[key] = updated
    effective_bytes = (json.dumps(effective, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    provenance = dict(source=str(source), source_sha256=sha256(source_bytes),
                      template=str(template), template_sha256=sha256(template_bytes),
                      effective_config=str(output), effective_sha256=sha256(effective_bytes),
                      model=str(model), model_sha256=sha256(model.read_bytes()), changes=changes,
                      limits='Python-visible settings only; scenario, bot seat selection and UI mode must be verified separately.')
    # Refuse sources that changed while the effective batch was being prepared.
    if source.read_bytes() != source_bytes or template.read_bytes() != template_bytes:
        raise ValueError('Source or template changed during preparation; retry after UI configuration completes')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(effective_bytes)
    with provenance_path.open('xb') as stream:
        stream.write((json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return {**provenance, 'provenance_path': str(provenance_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    parser.add_argument('--pythoncode-root', type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.out, template=args.template,
                         cssim_root=args.cssim_root, pythoncode_root=args.pythoncode_root)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Configuration preparation refused: {exc}\n')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
