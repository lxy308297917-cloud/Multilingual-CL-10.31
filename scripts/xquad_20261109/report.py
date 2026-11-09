"""Independently recompute XQuAD official EM/F1 and passage-paired intervals."""
from pathlib import Path
import csv,datetime,hashlib,itertools,json,sys,re,statistics
import numpy as np
from transformers.data.metrics.squad_metrics import compute_exact,compute_f1,normalize_answer
R=Path('/path/to/project');C=R/'configs/xquad_20261109';Q=R/'research/xquad_20261109';L=Path('/path/to/logs/qwen3-multilingual/xquad-20261109')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):
 if data:
  with (Q/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 models=load(C/'models.json');vectors={};scores=[];sources=[];missing=[];format_diagnostics={}
 for name,m in models.items():
  out=L/name
  if not (out/'DONE.json').exists():missing.append(name);continue
  files=list(out.rglob('samples_mcl_xquad_'+m['language']+'_*.jsonl'));assert len(files)==1;file=files[0];xs=[json.loads(x) for x in file.open()];assert len(xs)==1190;v={}
  for x in xs:
   assert x['filter']=='none';prediction=x['filtered_resps'][0];assert isinstance(prediction,str);refs=x['doc']['answers']['text'];em=max(compute_exact(a,prediction) for a in refs);f1=max(compute_f1(a,prediction) for a in refs);assert abs(em-x['exact_match'])<1e-10 and abs(f1-x['f1'])<1e-10;key=x['doc']['id'];assert key not in v;v[key]=dict(identity=(x['doc_hash'],x['prompt_hash'],x['target_hash']),cluster=x['doc']['paragraph_sha256'],EM=100*em,F1=100*f1)
  preds=[x['filtered_resps'][0] for x in xs];prefix=re.compile(r'^\s*(?:إجابة|الإجابة|جواب السؤال|Ответ|Ответ на вопрос|Answer)\s*[:：]',re.IGNORECASE)
  format_diagnostics[name]=dict(samples=1190,empty_outputs=sum(not p.strip() for p in preds),think_tag_outputs=sum('<think>' in p for p in preds),anchored_answer_label_prefix=sum(bool(prefix.match(p)) for p in preds),bold_markdown_outputs=sum('**' in p for p in preds),full_reference_token_containment=sum(any((' '+normalize_answer(a)+' ') in (' '+normalize_answer(x['filtered_resps'][0])+' ') for a in x['doc']['answers']['text']) for x in xs),median_prediction_reference_character_ratio=statistics.median(len(x['filtered_resps'][0])/max(1,len(x['doc']['answers']['text'][0])) for x in xs),scope='Posthoc output-format diagnostic prompted by exposed AR Base outputs. Frozen official EM/F1 unchanged; reference containment is not accuracy and may count copying/context repetition; no answer extraction, new prompt or test-based scoring choice.')
  results=list(out.rglob('results_*.json'));assert len(results)==1;aggregate=load(results[0])['results']['mcl_xquad_'+m['language']];row=dict(language=m['language'],model=name,role=m['role'],samples=1190,paragraph_clusters=len({x['cluster'] for x in v.values()}),EM=float(np.mean([x['EM'] for x in v.values()])),F1=float(np.mean([x['F1'] for x in v.values()])));assert row['paragraph_clusters']==240;assert abs(row['EM']/100-aggregate['exact_match,none'])<1e-10 and abs(row['F1']/100-aggregate['f1,none'])<1e-10;scores.append(row);vectors[name]=v;sources.append(dict(model=name,samples_sha256=sha(file),results_sha256=sha(results[0]),checkpoint_sha256=load(out/'DONE.json')['model_sha256']))
  (Q/'numeric').mkdir(exist_ok=True);ids=sorted(v);(Q/'numeric'/(name+'.json')).write_text(json.dumps(dict(model=name,language=m['language'],item_ids=ids,input_identity_sha256=[hashlib.sha256(json.dumps(v[k]['identity']).encode()).hexdigest() for k in ids],clusters=[v[k]['cluster'] for k in ids],EM=[v[k]['EM'] for k in ids],F1=[v[k]['F1'] for k in ids]),separators=(',',':'))+'\n')
 paired=[]
 for a,b in itertools.combinations(vectors,2):
  if models[a]['language']!=models[b]['language']:continue
  av,bv=vectors[a],vectors[b];assert av.keys()==bv.keys() and all(av[k]['identity']==bv[k]['identity'] and av[k]['cluster']==bv[k]['cluster'] for k in av);keys=sorted(av);clusters,inverse=np.unique([av[k]['cluster'] for k in keys],return_inverse=True);n=len(clusters);counts=np.bincount(inverse,minlength=n)
  for metric in ['EM','F1']:
   differences=np.array([bv[k][metric]-av[k][metric] for k in keys]);sums=np.bincount(inverse,weights=differences,minlength=n);rng=np.random.default_rng(42);draws=[]
   for _ in range(40):
    ix=rng.integers(n,size=(50,n));draws.extend(sums[ix].sum(1)/counts[ix].sum(1))
   lo,hi=np.quantile(draws,[.025,.975]);paired.append(dict(language=models[a]['language'],reference=a,model=b,metric=metric,delta=float(differences.mean()),ci95_low=float(lo),ci95_high=float(hi),samples=1190,paragraph_clusters=240,resamples=2000))
 (Q/'format_diagnostics.json').write_text(json.dumps(format_diagnostics,indent=2)+'\n')
 write('scores.csv',scores);write('paired_changes.csv',paired);(Q/'score_audit.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),completed_models=len(scores),required_models=len(models),missing=missing,all_raw_EM_F1_recomputed=True,all_official_references_used=True,paired_unit='240 original passages, all questions retained together',sources=sources,not_training_seed_uncertainty=True),indent=2)+'\n')
 md=['# XQuAD补充结果','',datetime.datetime.now().astimezone().isoformat(),'','AR、RU各1190题、240段文章。SW不受官方支持，标记UNSUPPORTED。XQuAD保留为补充，不加入固定5+5平均。主评分使用官方SQuAD v1.1英文规范化、参考答案最大EM/F1，输出原文直接评分，不按测试成绩选答案提取方式。','','| 语言 | 模型 | EM% | F1% |','|---|---|---:|---:|'];lookup={x['model']:x for x in scores}
 for name,m in models.items():md.append('| '+m['language']+' | '+name+' | '+' | '.join('—' if name not in lookup else f'{lookup[name][t]:.2f}' for t in ['EM','F1'])+' |')
 md+=['','输出格式诊断见format_diagnostics.json。带答案标签、额外单位或完整句的输出会受到严格EM/F1惩罚，不能将低EM直接等同于不会阅读。Markdown符号本身会被官方ASCII标点规则去除，额外词语仍保留。诊断不改变原始评分或提示。','','—为尚未完成；四个尚未训练且按用户要求暂缓的模型另见deferred_models.json，未用其他模型冒充。按文章分簇配对区间见paired_changes.csv，仅反映固定模型样本不确定性。没有新增SQuAD训练，不与论文中SQuAD专门训练后的抽取式模型直接排名。',''];(Q/'results.md').write_text('\n'.join(md));print(json.dumps(dict(completed=len(scores),required=len(models),paired_intervals=len(paired))))
if __name__=='__main__':main()
