"""Full-model SFT with one-backward representation preservation and CPUAdam."""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,os,shutil,sys
import torch,yaml
from safetensors.torch import load_file
R=Path('/path/to/project')
D=Path('/path/to/storage/datasets/functional-representation-20261109/p2-v1')
# Import our joint loss BEFORE adding the historic accounting directory.
from joint_loss import trainer_class
sys.path[:0]=[str(R/'scripts/sft_v1'),str(R/'scripts')]
from train_formal import ExposureAccounting
from train_resume_safe import OptimizerBinding

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).open()]
class Audit(ExposureAccounting):
 def __init__(self,holder,steps,mode):super().__init__();self.holder=holder;self.steps=steps;self.mode=mode;self.handles=[];self.gradients={}
 def on_train_begin(self,args,state,control,model=None,**kwargs):
  super().on_train_begin(args,state,control,model=model,**kwargs)
  assert args.world_size==1 and args.per_device_train_batch_size==1 and args.gradient_accumulation_steps==16 and state.max_steps==self.steps
  for layer in model.model.layers:
   function=layer._gradient_checkpointing_func
   while hasattr(function,'__wrapped__'):function=function.__wrapped__
   assert layer.gradient_checkpointing and function.keywords['use_reentrant'] is False,'HF re-enabled reentrant checkpointing'
  parameters={n:p for n,p in model.named_parameters() if p.requires_grad};assert len(parameters)==len(list(model.named_parameters()))
  assert self.parameters['trainable_unique_parameters']==self.parameters['unique_model_parameters']==4022468096
  assert all('lora_' not in n for n in parameters)
  for name,p in parameters.items():
   def hook(g,name=name):
    if name not in self.gradients:
     norm=torch.linalg.vector_norm(g,dtype=torch.float32);assert torch.isfinite(norm)
     self.gradients[name]=float(norm)
   self.handles.append(p.register_hook(hook))
  self.expected_names=set(parameters)
 def on_step_end(self,args,state,control,**kwargs):
  super().on_step_end(args,state,control,**kwargs)
  if state.global_step==1:
   assert set(self.gradients)==self.expected_names,'Missing full-parameter gradient hooks'
   for handle in self.handles:handle.remove()
   self.handles=[]
   p=Path(args.output_dir);p.mkdir(parents=True,exist_ok=True)
   (p/'gradient_coverage.json').write_text(json.dumps(dict(all_expected_parameters_received_finite_gradient=True,parameter_tensors=len(self.gradients),gradient_norms=self.gradients),indent=2)+'\n')
  control.should_save=state.global_step in ([self.steps] if self.steps<400 else [100,200,300,400]);return control
 def record(self,args,state,path):
  super().record(args,state,path);trainer=self.holder['trainer']
  (path/'old_exposure.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trainer.old_trace))
  (path/'joint_backward_audit.json').write_text(json.dumps(dict(target_microbatches=trainer.backend_backward_calls,backend_backward_calls=trainer.backend_backward_calls,mode=self.mode,expected_single_backward_per_target_microbatch=True,old_samples=len(trainer.old_trace),old_class_counts=dict(collections.Counter(x['capability'] for x in trainer.old_trace)),gradient_accumulation=16,old_samples_per_optimizer_step=0 if self.mode=='none' else 8,regularizer_coefficient=1,feature_layers=36,non_reentrant_checkpointing=True),indent=2)+'\n')
 def on_train_end(self,args,state,control,**kwargs):
  super().on_train_end(args,state,control,**kwargs);trainer=self.holder['trainer']
  assert state.global_step==self.steps and self.counts['samples']==self.steps*16
  assert trainer.backend_backward_calls==self.steps*16
  assert len(trainer.old_trace)==(0 if self.mode=='none' else self.steps*8)
  for handle in self.handles:handle.remove()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--mode',choices=['none','full36'],required=True);a=ap.parse_args();cfg=yaml.safe_load(a.config.read_text());out=Path(cfg['output_dir']);assert not out.exists()
 assert cfg['finetuning_type']=='full' and cfg['learning_rate']==1e-5 and cfg['use_reentrant_gc'] is False and cfg['gradient_checkpointing_kwargs']=={'use_reentrant':False}
 assert cfg['per_device_train_batch_size']==1 and cfg['gradient_accumulation_steps']==16 and cfg['seed']==42 and cfg['cutoff_len']==2048 and not cfg['packing']
 assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['AYA_EXPECTED_GPU_UUID']
 assert str(torch.cuda.get_device_properties(0).uuid)==os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-')
 lock=json.loads((R/'configs/aya_4b_20261105/data.lock.json').read_text());assert lock['rows']==6400
 assert all(sha(Path(cfg['dataset_dir'])/k)==v for k,v in lock['files'].items())
 manifest=json.loads((D/'manifest.json').read_text());assert all(sha(D/k)==v for k,v in manifest['files'].items())
 old=rows(D/'build_pool.jsonl');assert len(old)==128 and all(x['split']=='build' for x in old)
 expected=int(cfg['max_steps'])
 if expected<400:old=sorted(old,key=lambda x:-len(x['input_ids']))
 from deepspeed.ops.adam import cpu_adam
 lib=Path('/path/to/storage/torch-extensions-qwen3/cpu_adam/cpu_adam.so');spec=importlib.util.spec_from_file_location('cpu_adam',lib);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);cpu_adam.CPUAdamBuilder.load=lambda self:module
 import llamafactory.train.sft.workflow as workflow
 base=trainer_class(workflow.CustomSeq2SeqTrainer);holder={}
 class Configured(base):
  def __init__(self,*aa,**kk):
   super().__init__(*aa,**kk);self.protection_mode=a.mode;self.old_rows=old;self.normalizers=manifest['normalizers'];self.references=load_file(str(D/'build_reference_features.safetensors')) if a.mode=='full36' else None
   assert self.model.config.attention_dropout==0
   assert not any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in self.model.modules())
   holder['trainer']=self
 workflow.CustomSeq2SeqTrainer=Configured
 from llamafactory.train.tuner import run_exp
 start=datetime.datetime.now().astimezone().isoformat();run_exp(cfg,callbacks=[Audit(holder,expected,a.mode),OptimizerBinding(None)])
 stats=json.loads((out/'accounting-rank0.json').read_text());lr=rows(out/'applied-lr-rank0.jsonl');assert len(lr)==expected
 assert all(x['actual_lrs']==x['scheduler_last_lrs'] for x in lr)
 binding=json.loads((out/'optimizer-binding.json').read_text());assert binding['actual_optimizer_type']=='DeepSpeedCPUAdam' and binding['shared_group_dictionaries_after']
 trace=rows(out/'old_exposure.jsonl');target=rows(out/'exposure-rank0.jsonl');assert len(target)==expected*16
 if expected==400:
  source=rows('/path/to/storage/checkpoints/aya_4b_20261105/fft_final_v2/exposure-rank0.jsonl');assert [x['input_label_sha256'] for x in target]==[x['input_label_sha256'] for x in source]
  if a.mode=='full36':assert [x['id'] for x in trace]==[old[i%128]['id'] for i in range(3200)]
 else:
  assert a.mode=='none' or any(x['old_objective']>1e-10 for x in trace),'Engineering run must reach nonzero representation loss; no benchmark filter'
 shutil.copy2(Path(cfg['model_name_or_path'])/'generation_config.json',out/'generation_config.json')
 result=dict(status='TRAINING_DONE' if expected==400 else 'ENGINEERING_SMOKE_DONE',mode=a.mode,steps=expected,exposure=stats['global_counts'],elapsed_seconds=stats['elapsed_this_process_seconds'],peak_allocated_bytes=stats['max_memory_allocated'],weights_sha256={p.name:sha(p) for p in out.glob('*.safetensors')},config_sha256=sha(a.config),source_old_manifest_sha256=sha(D/'manifest.json'),target_order_matches_historical_fft=expected==400,old_samples=len(trace),old_check_samples=0,old_positions='P2 max4 reference-answer prediction positions; original raw36 block outputs',normalizers=manifest['normalizers'],lambda_repr=1,backward_calls=holder['trainer'].backend_backward_calls,applied_cpuadam_updates=len(lr),non_reentrant_checkpointing=True,saved_optimizer_state=True,started_at=start,ended_at=datetime.datetime.now().astimezone().isoformat(),final_policy='Fixed Final400, no performance gate or English early stop')
 (out/('TRAINING_DONE.json' if expected==400 else 'SMOKE_DONE.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
