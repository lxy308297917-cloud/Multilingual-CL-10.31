"""Rebuild the seven-track prerequisite from local, regenerated evidence."""
import json
from pathlib import Path
from datasets import load_from_disk
from common import ROOT, DATA, MODEL, sha, dump
from runtime import runtime_path

def main():
    tracks = {}
    for track in ['massive','mathoctopus','mtop','xnli','xlsum','wikilingua','sib200']:
        folder = DATA/('prepared-v2' if track == 'mtop' else 'prepared-v1')/track
        protocol = json.loads((folder/'protocol.json').read_text())
        expected = 640 if track == 'sib200' else 3200
        assert protocol['n_train_per_language'] == expected
        assert protocol['steps']*16 == expected
        assert protocol['model_revision'] == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
        assert protocol['tokenizer_sha256'] == sha(MODEL/'tokenizer.json')
        for name, digest in protocol['files'].items():
            assert sha(folder/name) == digest, ('Changed prepared source',track,name)
        cache = {}
        for lang in ['es','th','fr','en']:
            ds = load_from_disk(str(folder/(lang+'_tokenized')))['train']
            assert len(ds) == expected
            inputs = targets = 0
            for row in ds:
                ids, labels = row['input_ids'], row['labels']
                assert len(ids) == len(labels) <= 2048
                supervised = sum(value != -100 for value in labels[1:])
                assert supervised > 0 and -100 in labels
                assert all(a == b for a,b in zip(ids,labels) if b != -100)
                inputs += len(ids)
                targets += supervised
            totals = protocol['token_stats'][lang+':train']['totals']
            assert [inputs,targets] == totals[-2:], (track,lang,'Token accounting differs')
            cache[lang] = {'rows':len(ds),'input_tokens':inputs,'shifted_assistant_tokens':targets}
        tracks[track] = {'protocol_sha256':sha(folder/'protocol.json'),'training_examples_per_language':expected,'steps':protocol['steps'],'verified_cache':cache}
    output = Path(runtime_path('configs'))/'P0_GATE.json'
    assert not output.exists(), 'Never overwrite a prior gate'
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ROOT/'configs/unified_language_cl/SUMMARY_SCORING_CORRECTION_v2.json'
    destination = output.parent/policy.name
    assert policy.exists(), 'Missing historical summary-scoring correction policy'
    if destination.exists():
        assert sha(destination) == sha(policy)
    else:
        destination.write_bytes(policy.read_bytes())
    dump(output, {'status':'ALL_SEVEN_AUDITED','scope':'Reproduction gate: prepared source hashes, official model/tokenizer identity, fixed sample/step budgets and all 28 training caches. This is not independent linguistic quality certification.','tracks':tracks,'gpu_execution':False})
    print(json.dumps({'status':'ALL_SEVEN_AUDITED','tracks':7,'training_caches':28}))

if __name__ == '__main__':
    main()
