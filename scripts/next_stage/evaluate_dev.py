"""Identical frozen development MCQ/heldout NLL at every saved BF16 export."""
from pathlib import Path
import argparse,hashlib,json,os,collections,time,datetime
import torch
from transformers import AutoModelForCausalLM
from metrics import option_log_probabilities,summarize_options,shifted_assistant_nll

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 for n in ['model','data','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--label',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.time()
 def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES'];assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';protocol=json.loads((a.data/'protocol.json').read_text());assert not protocol['checkpoint_selection']
  for n,h in protocol['files_sha256'].items():assert sha(a.data/n)==h,n
  if a.step:
   state=json.loads((a.model/'trainer_state.json').read_text());assert state['global_step']==a.step and state['max_steps']==400
  weights=sorted(a.model.glob('*.safetensors'));assert weights
  dump('launch.json',{'label':a.label,'step':a.step,'model':str(a.model),'weights_sha256':{f.name:sha(f) for f in weights},'protocol_sha256':sha(a.data/'protocol.json'),'script_sha256':sha(Path(__file__)),'metrics_sha256':sha(Path(__file__).with_name('metrics.py')),'started_at':datetime.datetime.now().astimezone().isoformat(),'arithmetic':protocol['arithmetic'],'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
  torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).to('cuda:0').eval();model.config.use_cache=False;torch.cuda.reset_peak_memory_stats();group=collections.defaultdict(list)
  rows=[json.loads(l) for l in (a.data/'mcq.jsonl').open()]
  with torch.inference_mode(),(a.output/'mcq_predictions.jsonl').open('x') as out:
   for i,row in enumerate(rows):
    scores=option_log_probabilities(model,row['context_ids'],row['continuations'],'cuda:0');s=summarize_options(scores,row['gold']);v={'task':row['task'],'id':row['id'],'language':row['language'],'gold':row['gold'],'option_log_probabilities':scores,**s};group[row['task'],row['language']].append(v);out.write(json.dumps(v)+'\n')
    if (i+1)%100==0:out.flush();dump('progress.json',{'phase':'MCQ','completed':i+1,'total':len(rows),'seconds':time.time()-start})
  summaries={}
  for (task,lang),vals in group.items():
   n=len(vals);summaries[task+'|'+lang]={'n':n,'correct':sum(x['correct'] for x in vals),'acc':sum(x['correct'] for x in vals)/n,**{k:sum(x[k] for x in vals)/n for k in ['correct_answer_sequence_log_probability','correct_answer_4option_log_probability','gold_minus_best_wrong_margin']},'predicted_label_counts':dict(collections.Counter('ABCD'[x['predicted']] for x in vals)),'gold_label_counts':dict(collections.Counter('ABCD'[x['gold']] for x in vals)),'conditional_accuracy_by_gold':{g:sum(x['correct'] for x in vals if x['gold']==i)/max(1,sum(x['gold']==i for x in vals)) for i,g in enumerate('ABCD')}}
  total_loss=0.;total_tokens=0;nr=[json.loads(l) for l in (a.data/'heldout_nll.jsonl').open()]
  with torch.inference_mode(),(a.output/'nll_predictions.jsonl').open('x') as out:
   for i,v in enumerate(nr):
    ids=torch.tensor([v['input_ids']],device='cuda:0');labels=torch.tensor([v['labels']],device='cuda:0');logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits;loss,n=shifted_assistant_nll(logits,labels);total_loss+=loss;total_tokens+=n;out.write(json.dumps({'id':v['id'],'nll_sum':loss,'assistant_tokens':n})+'\n');del logits
    if (i+1)%64==0:out.flush();dump('progress.json',{'phase':'NLL','completed':i+1,'total':len(nr),'seconds':time.time()-start})
  assert total_tokens==protocol['nll_shifted_assistant_tokens']
  dump('DONE.json',{'status':'EXPOSED_DIAGNOSTIC_COMPLETE','label':a.label,'step':a.step,'mcq':summaries,'heldout_ha_nll':{'rows':len(nr),'nll_sum':total_loss,'assistant_tokens':total_tokens,'token_weighted_nll':total_loss/total_tokens},'protocol_sha256':sha(a.data/'protocol.json'),'prediction_sha256':sha(a.output/'mcq_predictions.jsonl'),'nll_prediction_sha256':sha(a.output/'nll_predictions.jsonl'),'seconds':time.time()-start,'gpu_seconds_occupied_including_loading':time.time()-start,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'formal_test_used_for_selection':False,'not_an_independent_confirmation':True})
 except BaseException as e:dump('FAILED.json',{'error':repr(e),'seconds':time.time()-start});raise
if __name__=='__main__':main()
