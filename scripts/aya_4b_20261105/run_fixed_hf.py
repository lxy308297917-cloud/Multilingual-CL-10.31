"""Run one frozen full HF benchmark group; never select a checkpoint."""
import argparse,subprocess,os,json,datetime
from pathlib import Path
from profile_paths import resolve

def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['belebele','gmmlu','mmlu'],required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--max-user-gpus',type=int,default=4);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu_uuid;assert not a.output.exists();root=Path(resolve('ROOT'));python=resolve('TRAIN_PY')
 subprocess.run([python,str(root/'scripts/preflight_gpu.py'),'--uuid',a.gpu_uuid,'--max-user-gpus',str(a.max_user_gpus),'--settle-seconds','20'],check=True)
 names,shots,batch={'belebele':('belebele_eng_Latn_locked,belebele_swh_Latn_locked',0,4),'gmmlu':('global_mmlu_full_sw_locked',5,1),'mmlu':('mmlu_locked',5,4)}[a.group];args=f'pretrained={a.model},tokenizer={a.tokenizer},dtype=bfloat16,enable_thinking=False'
 if a.adapter:args+=f',peft={a.adapter}'
 cmd=[str(Path(python).parent/'lm-eval'),'run','--model','hf','--model_args',args,'--tasks',names,'--include_path',str(a.tasks),'--num_fewshot',str(shots),'--apply_chat_template','--batch_size',str(batch),'--device','cuda:0','--output_path',str(a.output/'results'),'--log_samples','--seed','42'];a.output.mkdir(parents=True);(a.output/'command.json').write_text(json.dumps({'command':cmd,'fixed_budget_final_only':True,'started_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
 with (a.output/'eval.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 (a.output/'DONE.json').write_text(json.dumps({'status':'COMPLETE','group':a.group,'expected_samples':1800 if a.group=='belebele' else 14042})+'\n')
if __name__=='__main__':main()
