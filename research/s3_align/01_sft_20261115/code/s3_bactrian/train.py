"""Full historical Bactrian SFT plus once-per-update retention/alignment."""
import argparse
import collections
import contextlib
import datetime
import gc
import importlib.util
import os
import sys
import time
from common import *


@contextlib.contextmanager
def eval_grad(model):
    old=[(m,m.training) for m in model.modules()]
    model.eval()
    try:yield
    finally:
        for m,flag in old:m.training=flag


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['B','S','D','E'],required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    import torch,yaml
    from safetensors.torch import load_file
    sys.path.insert(0,str(MATH))
    from keep_loss import load_keep_assets,uniform_keep_loss
    sys.path.insert(0,'/srv/research/10.31/multilingual-research/scripts/sft_v1')
    from train_formal import ExposureAccounting
    cfg_path=J/'configs'/f'{a.arm}{"-smoke" if a.smoke else ""}.yaml'
    cfg=yaml.safe_load(cfg_path.read_text());out=Path(cfg['output_dir'])
    assert not out.exists(),'Never overwrite an attempt'
    for p,digest in read(J/'EXECUTION_LOCK.json').items():assert sha(p)==digest,p
    assert cfg['per_device_train_batch_size']==1 and cfg['gradient_accumulation_steps']==16
    assert cfg['num_train_epochs']==1 and not cfg.get('adapter_name_or_path') and not cfg.get('resume_from_checkpoint')
    keep=load_keep_assets() if a.arm!='S' else None
    rows=None;teacher=None;meta=None
    if a.arm!='B':
        done=read(CACHE/'DONE.json');assert done['status']=='BACTRIAN_INSTRUCTION_TEACHER_COMPLETE'
        assert all(sha(CACHE/p)==digest for p,digest in done['files'].items())
        assert done['input_manifest_sha256']==sha(J/'prepared/manifest.json')
        rows=[json.loads(l) for l in (J/'prepared/alignment.jsonl').open()]
        teacher=load_file(str(CACHE/'teacher.safetensors'));meta=read(CACHE/'metadata.json')
        assert [r['id'] for r in rows]==meta['ordered_ids']
    expected=4 if a.smoke else 4089
    holder={};started=time.monotonic()
    import llamafactory.train.sft.workflow as workflow
    original=workflow.CustomSeq2SeqTrainer
    class Joint(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);holder['trainer']=self
            self.last_aux_step=-1;self.aux_trace=[];self.alignment_probe=None
        def training_step(self,model,inputs,num_items_in_batch=None):
            assert self.accelerator.gradient_accumulation_steps==1
            result=super().training_step(model,inputs,num_items_in_batch=num_items_in_batch)
            step=int(self.state.global_step)
            if self.last_aux_step==step:return result
            self.last_aux_step=step
            record=dict(step=step+1,old_id=None,old_input_tokens=0,
                        keep_loss=0.0,keep_weight=0.0,auxiliary_exposures_per_update=1,
                        accelerator_GA=1,auxiliary_backward_divisor=1)
            total=result
            if keep is not None:
                # Exactly one English example per optimizer update, including
                # the last partial target batch. No extra GA division applies.
                with self.compute_loss_context_manager():old,info=uniform_keep_loss(model,keep,step)
                self.accelerator.backward(old)
                record.update(old_id=info['id'],old_input_tokens=info['input_tokens'],
                              keep_loss=float(old.detach()),keep_weight=1.0)
                total=total+old.detach()
                del old
            if a.arm!='B':
                index=step%len(rows);r=rows[index]['encoded']['sw']
                ids=torch.tensor([r['input_ids']],device=model.device)
                mode='random' if a.arm=='D' else 'shared'
                with eval_grad(model),self.compute_loss_context_manager():
                    h=pool(model,ids,r['body_positions'])
                    target=teacher['targets'][index].to(h.device)
                    basis=teacher[mode].to(h.device)
                    raw=((h-target)@basis).square().mean()
                    loss=.1*raw/meta['scales'][mode]
                    assert h.requires_grad and loss.requires_grad and torch.isfinite(loss)
                    if a.smoke and self.alignment_probe is None:
                        params=[p for p in model.parameters() if p.requires_grad]
                        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                        norm=sum(float(g.float().square().sum()) for g in grads if g is not None)**.5
                        assert norm>0 and all(torch.isfinite(g).all() for g in grads if g is not None)
                        self.alignment_probe=dict(nonzero_alignment_gradient_norm=norm,
                                                  teacher_requires_grad=target.requires_grad,
                                                  basis_requires_grad=basis.requires_grad)
                        del grads
                    self.accelerator.backward(loss)
                record.update(alignment_id=rows[index]['id'],alignment_mode=mode,
                              alignment_loss=float(loss.detach()),raw_alignment=float(raw.detach()),
                              alignment_input_tokens=len(r['input_ids']))
                total=total+loss.detach()
            self.aux_trace.append(record)
            return total
    class Counter(ExposureAccounting):
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            super().on_train_begin(args,state,control,model=model,**kwargs)
            assert args.world_size==1 and state.max_steps==expected
            assert self.parameters['trainable_unique_parameters']==33030144
            model.save_pretrained(out/'checkpoint-0',safe_serialization=True)
            torch.cuda.reset_peak_memory_stats()
        def record(self,args,state,path):
            super().record(args,state,path)
            (path/'auxiliary_exposure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in holder['trainer'].aux_trace))
    workflow.CustomSeq2SeqTrainer=Joint
    from llamafactory.train.tuner import run_exp
    run_exp(cfg,callbacks=[Counter()])
    trainer=holder['trainer'];stats=read(out/'accounting-rank0.json')
    assert stats['step']==expected and stats['trace_covers_all_local_samples']
    actual=[json.loads(l) for l in (out/'exposure-rank0.jsonl').open()]
    original_trace=K.parent/'bactrian_single_20261112/sw_final_v1/exposure-rank0.jsonl'
    original_rows=[json.loads(l) for l in original_trace.open()]
    assert [r['input_label_sha256'] for r in actual]==[r['input_label_sha256'] for r in original_rows[:len(actual)]], 'Changed main SFT example order'
    assert len(actual)==(64 if a.smoke else 65415)
    if not a.smoke:
        assert stats['global_counts']['input_tokens']==19135489
        assert stats['global_counts']['shifted_assistant_tokens']==14314363
    assert len(trainer.aux_trace)==expected
    if a.arm=='S':
        assert all(r['keep_weight']==0 and r['old_input_tokens']==0 and r['old_id'] is None for r in trainer.aux_trace)
    weights=load_file(str(out/'adapter_model.safetensors'));initial=load_file(str(out/'checkpoint-0/adapter_model.safetensors'))
    assert weights.keys()==initial.keys() and all(torch.isfinite(v).all() for v in weights.values())
    assert sum(not torch.equal(v,initial[k]) for k,v in weights.items())>200
    done=dict(status='SMOKE_TRAINED_PENDING_FRESH_RELOAD' if a.smoke else 'TRAINING_DONE',
        arm=a.arm,steps=expected,samples=len(actual),weights_sha256={'adapter_model.safetensors':sha(out/'adapter_model.safetensors')},
        main_sft_order_matches_historical_baseline=True,accounting=stats,
        alignment_gradient_probe=trainer.alignment_probe,auxiliary_rows=len(trainer.aux_trace),
        old_total_input_tokens=sum(r['old_input_tokens'] for r in trainer.aux_trace),
        alignment_total_input_tokens=sum(r.get('alignment_input_tokens',0) for r in trainer.aux_trace),
        elapsed_seconds=time.monotonic()-started,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        config_sha256=sha(cfg_path),execution_lock_sha256=sha(J/'EXECUTION_LOCK.json'))
    dump(out/('SMOKE_TRAINED.json' if a.smoke else 'TRAINING_DONE.json'),done)
    if not a.smoke:
        event=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   arm=a.arm,stage='training',status='TRAINING_COMPLETE_PENDING_FRESH_RELOAD',
                   steps=expected,samples=len(actual),path=str(out),
                   next='Fresh reload verification, then target-language benchmarks before English general-ability evaluation.')
        with (O/'MILESTONES.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    print(json.dumps({k:done[k] for k in ['status','arm','steps','samples','elapsed_seconds','gpu']}),flush=True)


if __name__=='__main__':main()
