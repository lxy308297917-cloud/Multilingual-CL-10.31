"""Pin the upstream lm-eval MMLU task prompts to exact dataset revisions.

Generated YAML stays outside Git; this script and the locked harness commit are
the source of truth. It copies upstream prompt/scorer definitions verbatim and
only appends dataset revision plus distinct task names and a weighted group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import lm_eval


ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / 'assets.lock.json').read_text())['assets']
TASKS = Path(lm_eval.__file__).parent / 'tasks'
SPECS = [
    ('mmlu', TASKS / 'mmlu/default', 'mmlu_*.yaml', '_default_template_yaml', ASSETS['mmlu_test']['revision'], 'mmlu_locked'),
    ('global_mmlu_full_sw', TASKS / 'global_mmlu/full/sw', 'global_mmlu_full_sw_*.yaml', '_sw_template_yaml', ASSETS['global_mmlu_sw_test']['revision'], 'global_mmlu_full_sw_locked'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists(), args.output_dir
    args.output_dir.mkdir(parents=True)
    report = {'harness_tasks_root': str(TASKS), 'groups': {}}
    for prefix, source, pattern, template, revision, group in SPECS:
        target = args.output_dir / prefix
        target.mkdir()
        source_files = sorted(source.glob(pattern))
        assert len(source_files) == 57, (prefix, len(source_files))
        template_text = (source / template).read_text()
        assert 'dataset_kwargs:' not in template_text
        (target / template).write_text(template_text.rstrip() + f'\ndataset_kwargs:\n  revision: {revision}\n')
        if (source / 'utils.py').exists():
            shutil.copy2(source / 'utils.py', target / 'utils.py')
        names = []
        for file in source_files:
            content = file.read_text()
            old_name = file.stem
            new_name = old_name + '_locked'
            content, changed = re.subn(r'(?m)^"?task"?: .*$', f'task: {new_name}', content)
            assert changed == 1, file
            for category in ('stem', 'other', 'social_sciences', 'humanities'):
                content = content.replace(f'{prefix}_{category}_tasks', f'{group}_{category}_tasks')
            (target / f'{new_name}.yaml').write_text(content)
            names.append(new_name)
        group_yaml = 'group: ' + group + '\ntask:\n' + ''.join('  - ' + name + '\n' for name in names) + 'aggregate_metric_list:\n  - metric: acc\n    weight_by_size: true\n'
        (target / f'_{group}.yaml').write_text(group_yaml)
        report['groups'][group] = {'tasks': len(names), 'dataset_revision': revision}
    report['file_sha256'] = {
        str(file.relative_to(args.output_dir)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(args.output_dir.rglob('*')) if file.is_file()
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['groups'], sort_keys=True))


if __name__ == '__main__':
    main()
