"""Upstream training callbacks: preserve historical supervision-budget counters."""
import hashlib,json,math,os,resource,time
from pathlib import Path
import torch
from transformers import TrainerCallback

class Accounting(TrainerCallback):
    def __init__(self, resume=None, stop_at=None):
        self.resume=resume; self.stop_at=stop_at; self.counts={'samples':0,'input_tokens':0,'shifted_assistant_tokens':0,'padded_positions':0}
        self.started=time.monotonic(); self.hook=None

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        if self.resume:
            p=Path(self.resume)/f'accounting-rank{args.process_index}.json'
            assert p.exists(), 'Resume must retain prior exact token counters'
            old=json.loads(p.read_text()); self.counts=old['local_counts']
            assert old['step']==state.global_step, (old['step'],state.global_step)
        self.initial_step=state.global_step
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=sum(p.numel() for p in model.parameters())
        self.parameters={'total_local_parameters':total,'trainable_local_parameters':trainable}
        def observe(module,inputs,kw):
            if not module.training:return
            ids=kw.get('input_ids'); labels=kw.get('labels'); mask=kw.get('attention_mask')
            assert ids is not None and labels is not None
            assert (labels==-100).any() and (labels[:,1:]!=-100).any(), 'SFT supervision mask invalid'
            self.counts['samples']+=ids.shape[0]
            self.counts['input_tokens']+=int(mask.sum()) if mask is not None else ids.numel()
            self.counts['shifted_assistant_tokens']+=int((labels[:,1:]!=-100).sum())
            self.counts['padded_positions']+=ids.numel()
        self.hook=model.register_forward_pre_hook(observe,with_kwargs=True)

    def record(self,args,state,path):
        values=torch.tensor(list(self.counts.values()),dtype=torch.int64,device=args.device)
        if torch.distributed.is_initialized():torch.distributed.all_reduce(values)
        result={'step':state.global_step,'resume_initial_step':self.initial_step,'local_counts':self.counts.copy(),'global_counts':dict(zip(self.counts,map(int,values.tolist()))),'elapsed_this_process_seconds':time.monotonic()-self.started,'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':self.parameters,'world_size':args.world_size,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'log_history':state.log_history}
        path.mkdir(parents=True,exist_ok=True)
        (path/f'accounting-rank{args.process_index}.json').write_text(json.dumps(result,indent=2)+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if self.stop_at and state.global_step>=self.stop_at:
            control.should_save=True;control.should_training_stop=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs and 'loss' in logs:
            import math
            if not math.isfinite(logs['loss']):raise RuntimeError('Nonfinite training loss')

    def on_save(self,args,state,control,**kwargs):
        self.record(args,state,Path(args.output_dir)/f'checkpoint-{state.global_step}')

    def on_train_end(self,args,state,control,**kwargs):
        self.record(args,state,Path(args.output_dir))
        if self.hook:self.hook.remove()

class ExposureAccounting(Accounting):
    def __init__(self,resume=None,stop_at=None):
        super().__init__(resume,stop_at);self.exposure=[];self.exposure_hook=None;self.loss_hook=None

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        super().on_train_begin(args,state,control,model=model,**kwargs)
        # ZeRO3 parameters have numel()==0 between gathers; ds_numel is global.
        self.parameters={'unique_model_parameters':sum(getattr(p,'ds_numel',p.numel()) for p in model.parameters()),'trainable_unique_parameters':sum(getattr(p,'ds_numel',p.numel()) for p in model.parameters() if p.requires_grad)}
        self.state_ref=state;self.micro=0
        if self.resume:
            p=Path(self.resume)/f'exposure-rank{args.process_index}.jsonl'
            if p.exists():self.exposure=[json.loads(line) for line in p.open()]
        def observe(module,inputs,kw):
            if not module.training:return
            ids=kw['input_ids'];labels=kw['labels'];mask=kw['attention_mask']
            for j in range(ids.shape[0]):
                keep=mask[j].bool();tokens=ids[j][keep].tolist();targets=labels[j][keep].tolist()
                sha=hashlib.sha256(json.dumps([tokens,targets],separators=(',',':')).encode()).hexdigest()
                self.exposure.append({'update_step':self.state_ref.global_step+1,'microstep':self.micro,'rank':args.process_index,'row_in_microbatch':j,'input_label_sha256':sha,'input_tokens':len(tokens),'shifted_assistant_tokens':sum(t!=-100 for t in targets[1:])})
            self.micro+=1
        self.exposure_hook=model.register_forward_pre_hook(observe,with_kwargs=True)
        def finite_loss(module,inputs,output):
            loss=getattr(output,'loss',None)
            if module.training and loss is not None and not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError('Nonfinite actual forward loss before backward')
        self.loss_hook=model.register_forward_hook(finite_loss)

    def on_step_begin(self,args,state,control,**kwargs):
        self.micro=0

    def record(self,args,state,path):
        super().record(args,state,path)
        p=path/f'exposure-rank{args.process_index}.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in self.exposure))
        f=path/f'accounting-rank{args.process_index}.json';j=json.loads(f.read_text());j['exposure_file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();j['trace_rows']=len(self.exposure);j['trace_covers_all_local_samples']=len(self.exposure)==self.counts['samples'];f.write_text(json.dumps(j,indent=2)+'\n')

    def on_train_end(self,args,state,control,**kwargs):
        super().on_train_end(args,state,control,**kwargs)
        if self.exposure_hook:self.exposure_hook.remove()
        if self.loss_hook:self.loss_hook.remove()
