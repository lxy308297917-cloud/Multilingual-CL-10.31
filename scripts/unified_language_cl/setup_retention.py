"""Create isolated local task YAMLs; never edit an installed environment."""
import json
import subprocess
from pathlib import Path
import yaml
from common import ROOT, sha
from runtime import runtime_path

def write_once(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Existing generated resource differs; choose a fresh MLCL namespace: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

def main():
    spec_path = ROOT/'configs/unified_language_cl/retention_tasks.lock.json'
    spec = json.loads(spec_path.read_text())
    sources = json.loads((ROOT/'configs/unified_language_cl/sources.lock.json').read_text())
    checked_data = {}
    for asset in sources['assets']:
        if asset['root'] in ['gsm8k_source', 'ifeval_source']:
            path = Path(runtime_path(asset['root']))/asset['path']
            assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256'], 'Missing or changed benchmark source: '+asset['path']
            checked_data[asset['root']+'/'+asset['path']] = asset['sha256']
    belebele = Path(runtime_path('belebele_source'))/'eng_Latn.jsonl'
    assert sha(belebele) == sources['derived_belebele_eng_Latn']['sha256']
    assert sum(1 for _ in belebele.open()) == 900
    checked_data['belebele_source/eng_Latn.jsonl'] = sha(belebele)
    probe = "import json,lm_eval; from pathlib import Path; print(json.dumps(str(Path(lm_eval.__file__).parent/'tasks')))"
    roots = {}
    for role in ['train_python', 'eval_python']:
        value = subprocess.check_output([runtime_path(role), '-c', probe], text=True)
        roots[role] = Path(json.loads(value.strip().splitlines()[-1]))
        for task in spec['tasks'].values():
            assert sha(roots[role]/task['upstream_include']) == task['upstream_include_sha256'], 'Install the pinned Harness source in both environments'
    # Existing experiments seeded the language detector at interpreter startup.
    # Check the same effective setting, without modifying the user's environment.
    seed = subprocess.check_output([runtime_path('eval_python'), '-c', 'from langdetect import DetectorFactory; print(DetectorFactory.seed)'], text=True).strip()
    if seed != str(spec['langdetect_seed']):
        raise ValueError('The evaluation environment must set langdetect.DetectorFactory.seed=42 at interpreter startup; see reproduction instructions')
    core = Path(runtime_path('retention_core_tasks'))
    reading = Path(runtime_path('retention_reading_tasks'))
    generated = {}
    for name, task in spec['tasks'].items():
        value = dict(task['definition'])
        value['include'] = str(roots['train_python']/task['upstream_include'])
        if name == 'gsm8k':
            value['dataset_kwargs'] = {'data_files':{split:str(Path(runtime_path('gsm8k_source'))/(split+'-00000-of-00001.parquet')) for split in ['train','test']}}
        else:
            value['dataset_kwargs'] = {'data_files':{'train':str(Path(runtime_path('ifeval_source'))/'ifeval_input_data.jsonl')}}
        path = core/(name+'.yaml')
        write_once(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        generated[path.name] = sha(path)
    write_once(reading/'_belebele_locked_base.yaml', spec['reading_template'])
    write_once(reading/'belebele_eng_Latn_locked.yaml', yaml.safe_dump({'include':'_belebele_locked_base.yaml','task':'belebele_eng_Latn_locked','dataset_kwargs':{'data_files':{'test':str(Path(runtime_path('belebele_source'))/'eng_Latn.jsonl')}}}, sort_keys=False))
    lock = {'tasks':generated,'data_sha256':checked_data,'harness_commit':spec['harness_commit'],'definition_lock_sha256':sha(spec_path),'langdetect_seed':42,'generated_paths_only':True}
    write_once(runtime_path('retention_lock'), json.dumps(lock, indent=2)+'\n')
    # The runtime evaluator also checks an isolated manifest. Only used native
    # reading resources are included; unrelated historical tasks are not needed.
    old = Path(runtime_path('old_eval_data'))
    write_once(old/'manifest.json', json.dumps({'file_sha256':{},'scope':'Three-task reproduction uses generated task locks and source lock, without unused historical task trees.'}, indent=2)+'\n')
    print(json.dumps({'status':'READY','core_tasks':len(generated),'reading_tasks':1,'source_environments_modified':False}))

if __name__ == '__main__':
    main()
