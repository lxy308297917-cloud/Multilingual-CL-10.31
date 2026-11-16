"""No-update endpoint check of the frozen initial gradient calibration."""
from common import *
from train_mapping import alignment,eval_grad,gradnorm
import argparse,torch,time,os,numpy as np
from transformers import AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from safetensors.torch import load_file
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['T1','T2'],required=True);a=ap.parse_args();out=R/'analysis'/f'final_gradient_{a.arm}.json';assert not out.exists()
 cp=CK/f'{a.arm}-full-v1';done=read(cp/'TRAINING_DONE.json');assert done['status']=='TRAINING_DONE' and done['steps']==4089
 digest=sha(cp/'adapter_model.safetensors');assert done['weights_sha256']['adapter_model.safetensors']==digest
 torch.set_num_threads(4);start=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
 model=PeftModel.from_pretrained(model,str(cp),is_trainable=True).eval();params=[p for p in model.parameters() if p.requires_grad];before=[p.detach().cpu().clone() for p in params]
 ds=load_from_disk(str(DATA/'sw_train_tokenized'))['train'];lookup={r['id']:i for i,r in enumerate(rows(DATA/'sw_train.jsonl'))};pool=rows(R/'inputs/alignment.jsonl');metadata=read(R/'assets/mapping_metadata.json');maps=load_file(str(R/'assets/mappings.safetensors'));cal=read(R/'assets/gradient_calibration.json');measurements=[]
 for i in metadata['calibration_indices']:
  row=pool[i];sample=ds[lookup[row['id']]]
  with eval_grad(model),torch.autocast('cuda',dtype=torch.bfloat16):
   batch={k:torch.tensor([sample[k]],device='cuda') for k in ['input_ids','attention_mask','labels']};loss=model(**batch,use_cache=False).loss;nce=gradnorm(loss,params);del loss,batch
   loss=alignment(model,row['encoded']['sw'],i,a.arm,maps);raw=float(loss.detach());naux=gradnorm(loss,params);del loss
  measurements.append(dict(id=row['id'],ce_gradient_norm=nce,raw_alignment_gradient_norm=naux,lambda_fixed=cal['lambda'][a.arm],weighted_alignment_gradient_norm=cal['lambda'][a.arm]*naux,weighted_over_ce=cal['lambda'][a.arm]*naux/nce,raw_alignment_loss=raw))
 assert all(torch.equal(p.detach().cpu(),b) for p,b in zip(params,before));assert sha(cp/'adapter_model.safetensors')==digest
 median_ce=float(np.median([r['ce_gradient_norm'] for r in measurements]));median_aux=float(np.median([r['weighted_alignment_gradient_norm'] for r in measurements]))
 write(out,dict(at=now(),arm=a.arm,steps=4089,rows=measurements,median_ce_norm=median_ce,median_weighted_aux_norm=median_aux,ratio_of_medians=median_aux/median_ce,initial_ratio_of_medians=.1,coefficient_unchanged=True,all_parameters_bitwise_unchanged=True,optimizer_updates=0,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.monotonic()-start,adapter_sha256=digest,interpretation='Same32 build examples and no-dropout convention as initial calibration. Endpoint comparison only; intermediate traces contain auxnorm only. Not actual AdamW update ratio.'))
 print(json.dumps({'arm':a.arm,'ratio_of_medians':median_aux/median_ce}))
if __name__=='__main__':main()
