"""Read-only final-weight geometry for completed Base->SW->HA FFT stages."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import torch
from safetensors import safe_open
p=argparse.ArgumentParser()
for k in ['base','stage1','stage2','initial','sst-code','identity-record','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.time()
sys.path.insert(0,str(a.sst_code));from projector import is_target
assert not torch.cuda.is_initialized();torch.set_num_threads(4);torch.set_grad_enabled(False)
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def mappings(root):
 out={}
 for path in root.glob('*.safetensors'):
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():assert key not in out;out[key]=path
 assert json.loads((root/'config.json').read_text())['tie_word_embeddings'] is True
 if 'lm_head.weight' in out:
  with safe_open(out['lm_head.weight'],framework='pt',device='cpu') as f:head=f.get_tensor('lm_head.weight')
  with safe_open(out['model.embed_tokens.weight'],framework='pt',device='cpu') as f:embedding=f.get_tensor('model.embed_tokens.weight')
  assert torch.equal(head,embedding),'Serialized tied aliases differ'
  del out['lm_head.weight']
 return out
maps=[mappings(x) for x in [a.base,a.stage1,a.stage2]];assert maps[0].keys()==maps[1].keys()==maps[2].keys()
identity=json.loads(a.identity_record.read_text());expected=identity['model_weight_sha256']
files={}
for root in [a.base,a.stage1,a.stage2]:
 for path in root.glob('*.safetensors'):
  digest=sha(path);assert digest==expected[str(root)][path.name];files[str(path)]=digest
initial=torch.load(a.initial,map_location='cpu',weights_only=False);assert len(initial['bases'])==196
rows=[]
for name in sorted(maps[0]):
 vals=[]
 for m in maps:
  with safe_open(m[name],framework='pt',device='cpu') as f:vals.append(f.get_tensor(name).float())
 w0,w1,w2=vals;old=w1-w0;new=w2-w1
 def moments(x,y):
  result=[0.,0.,0.]
  for off in range(0,x.numel(),4*1024**2):
   u=x.flatten()[off:off+4*1024**2].double();v=y.flatten()[off:off+4*1024**2].double()
   result[0]+=float(u.square().sum());result[1]+=float(v.square().sum());result[2]+=float((u*v).sum())
  return result
 aa,bb,ab=moments(old,new);total=aa+bb+2*ab
 item={'name':name,'elements':old.numel(),'target':is_target(name),'stage1_delta_squared_norm':aa,'stage2_increment_squared_norm':bb,'delta_inner_product':ab,'stage2_total_from_base_squared_norm':total,
  'cosine':ab/(aa*bb)**.5 if aa>0 and bb>0 else None,'old_direction_retained_coefficient':1+ab/aa if aa>0 else None}
 if item['target']:
  assert torch.equal(w0,initial['references'][name]),'Base reference mismatch'
  basis=initial['bases'][name]
  fo=basis.project(old);fn=basis.project(new);fa,fb,fd=moments(fo,fn)
  item['fixed_Base_free']={'stage1_delta_squared_norm':fa,'stage2_increment_squared_norm':fb,'delta_inner_product':fd,'cosine':fd/(fa*fb)**.5 if fa>0 and fb>0 else None,'old_direction_retained_coefficient':1+fd/fa if fa>0 else None}
 rows.append(item);del vals,w0,w1,w2,old,new
 def save_progress():
  (a.output/'progress.json').write_text(json.dumps({'completed':len(rows),'total':len(maps[0]),'seconds':time.time()-start})+'\n')
 if len(rows)%28==0:save_progress()
summary={}
for kind,selected in [('all',rows),('target196',[x for x in rows if x['target']]),('outside196',[x for x in rows if not x['target']])]:
 aa=sum(x['stage1_delta_squared_norm'] for x in selected);bb=sum(x['stage2_increment_squared_norm'] for x in selected);ab=sum(x['delta_inner_product'] for x in selected)
 summary[kind]={'elements':sum(x['elements'] for x in selected),'stage1_delta_squared_norm':aa,'stage2_increment_squared_norm':bb,'delta_inner_product':ab,'cosine':ab/(aa*bb)**.5,'old_direction_retained_coefficient':1+ab/aa,'final_vs_stage1_distance_to_base_ratio':((aa+bb+2*ab)/aa)**.5}
record={'status':'READ_ONLY_FINAL_GEOMETRY_COMPLETE','source_weights_sha256':files,'base_reference_all196_exact':True,'serialized_tied_aliases_verified_and_counted_once':True,'rows':rows,'summary':summary,'seconds':time.time()-start,
 'limitations':['Cumulative stage deltas, not per-step Adam trajectories.','Negative inner product means geometric opposition, not proof of forgetting.','Fixed Base spectrum used solely for geometry; no test selection.','BF16 saved weights promoted to FP32; not native optimizer masters.','Tied embedding/head represented once; no separation of input/output causal roles.']}
(a.output/'audit.json').write_text(json.dumps(record,indent=2)+'\n');(a.output/'DONE.json').write_text(json.dumps({'status':record['status'],'seconds':record['seconds']})+'\n');print(json.dumps(summary))
