"""Process-local LF Trainer extension. Native SFT/data/optimizer remain unchanged."""
import argparse,contextlib,hashlib,json,sys,time
from pathlib import Path
import torch,yaml
from losses_replay import auxiliary_loss, full_vocabulary_replay
AUX=False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--teacher',type=Path,required=True)
    ap.add_argument('--mode',choices=['marginal','content','replay'],required=True)
    ap.add_argument('--stop-at',type=int,default=100)
    a=ap.parse_args();sys.path.insert(0,str(a.project/'scripts/sft_v1'))
    from train_formal import ExposureAccounting
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    teacher=json.loads(a.teacher.read_text());assert teacher['selected_count']>=16
    pool={(r['id'],r['language']):r for r in map(json.loads,(a.data/'train_aux.jsonl').open()) if r['id'] in teacher['selected_ids'] and r['language'] in ['en','sw']}
    base=workflow.CustomSeq2SeqTrainer
    class PilotTrainer(base):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            global AUX
            result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
            if not model.training:return result
            step=self.state.global_step
            if getattr(self,'last_aux_step',None)==step:return result
            self.last_aux_step=step
            lang=['en','sw'][step%2];sid=teacher['selected_ids'][(step//2)%len(teacher['selected_ids'])]
            row=pool[sid,lang];parts=[];full_logits=[];gold_tokens=[];input_tokens=0;started=time.monotonic()
            # Match the actual HF4.57 training_step normalization, including loss kwargs.
            divided=(not self.model_accepts_loss_kwargs or kwargs.get('num_items_in_batch') is None) and self.compute_loss_func is None
            factor=self.current_gradient_accumulation_steps if divided else 1
            AUX=True
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    for r in row['rotations']:
                        pairs=r['letter_pairs'];c=pairs[0][0];prefix=pairs[0][1][:-1]
                        assert all(cc==c and t[:-1]==prefix for cc,t in pairs)
                        ids=torch.tensor([c+prefix],device=inputs['input_ids'].device)
                        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[0,-1]
                        parts.append(logits[[t[-1] for _,t in pairs]].float());full_logits.append(logits.float());gold_tokens.append(pairs[r['order'].index(row['gold'])][1][-1]);input_tokens+=ids.numel()
                    aux=full_vocabulary_replay(torch.stack(full_logits),gold_tokens) if a.mode=='replay' else auxiliary_loss(a.mode,torch.stack(parts),[r['order'] for r in row['rotations']],row['gold'],teacher['targets'][sid][lang],teacher['label_marginal'][lang])
            finally:AUX=False
            assert bool(torch.isfinite(aux)), 'nonfinite auxiliary objective'
            record=dict(update_step=step+1,semantic_id=sid,language=lang,auxiliary_loss=float(aux.detach()),input_tokens=input_tokens,
                rotations=4,replay_denominator='full vocabulary' if a.mode=='replay' else None,rank=self.args.process_index,hf_accumulation_factor=factor,gradient_accumulation_steps=self.current_gradient_accumulation_steps,
                model_accepts_loss_kwargs=self.model_accepts_loss_kwargs,num_items_present=kwargs.get('num_items_in_batch') is not None,
                host_dispatch_seconds=time.monotonic()-started)
            with (Path(self.args.output_dir)/f'aux-rank{self.args.process_index}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if return_outputs:return result[0]+aux*factor,result[1]
            return result+aux*factor
    class GuardedAccounting(ExposureAccounting):
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            super().on_train_begin(args,state,control,model=model,**kwargs)
            self.hook.remove();self.exposure_hook.remove();self.hook=None
            def observe(module,inputs,kw):
                if AUX or not module.training:return
                ids=kw['input_ids'];labels=kw['labels'];mask=kw['attention_mask']
                assert (labels==-100).any() and (labels[:,1:]!=-100).any()
                self.counts['samples']+=ids.shape[0];self.counts['input_tokens']+=int(mask.sum())
                self.counts['shifted_assistant_tokens']+=int((labels[:,1:]!=-100).sum());self.counts['padded_positions']+=ids.numel()
                for j in range(ids.shape[0]):
                    keep=mask[j].bool();tokens=ids[j][keep].tolist();targets=labels[j][keep].tolist()
                    digest=hashlib.sha256(json.dumps([tokens,targets],separators=(',',':')).encode()).hexdigest()
                    self.exposure.append(dict(update_step=self.state_ref.global_step+1,microstep=self.micro,rank=args.process_index,row_in_microbatch=j,input_label_sha256=digest,input_tokens=len(tokens),shifted_assistant_tokens=sum(t!=-100 for t in targets[1:])))
                self.micro+=1
            self.exposure_hook=model.register_forward_pre_hook(observe,with_kwargs=True)
    workflow.CustomSeq2SeqTrainer=PilotTrainer
    config=yaml.safe_load(a.config.read_text())
    assert config['gradient_accumulation_steps']==8 and config['learning_rate']==1e-5
    assert config['num_train_epochs']==1.0 and config['warmup_ratio']==0.05
    run_exp(config,callbacks=[GuardedAccounting(stop_at=a.stop_at)])

if __name__=='__main__':main()
