"""Call official LF SFT; record actual exposure and verify saved adapter."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import torch,yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel,get_peft_model_state_dict
ROOT=Path('/srv/research/10.31/multilingual-research')
sys.path.insert(0,str(ROOT/'scripts/sft_v1'))
from train_formal import ExposureAccounting
from llamafactory.train.tuner import run_exp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text());out=Path(cfg['output_dir'])
    assert not out.exists(), 'Preserve attempts; use a new output directory'
    assert not cfg.get('adapter_name_or_path') and not cfg.get('resume_from_checkpoint')
    assert cfg['num_train_epochs']==1 and cfg['seed']==42
    assert cfg['per_device_train_batch_size']*cfg['gradient_accumulation_steps']==16
    freeze=json.loads(a.config.with_suffix('.lock.json').read_text())
    for filename,expected in freeze.items():assert sha(filename)==expected,filename
    class TimedExposure(ExposureAccounting):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            return super().on_train_begin(args,state,control,**kwargs)
    counter=TimedExposure()
    run_exp(cfg,callbacks=[counter])
    stats=json.loads((out/'accounting-rank0.json').read_text())
    assert stats['world_size']==1 and stats['trace_covers_all_local_samples']
    weights=load_file(str(out/'adapter_model.safetensors'))
    assert weights and all(torch.isfinite(v).all().item() for v in weights.values())
    expected_steps=4 if a.smoke else (json.loads(Path(cfg['dataset_dir'],'manifest.json').read_text())['stats'][cfg['dataset']]['rows']+15)//16
    assert stats['step']==expected_steps,(stats['step'],expected_steps)
    if not a.smoke:
        assert stats['global_counts']['samples']==json.loads(Path(cfg['dataset_dir'],'manifest.json').read_text())['stats'][cfg['dataset']]['rows']
    receipt={'status':'SMOKE_TRAINED' if a.smoke else 'TRAINING_DONE','config_sha256':sha(a.config),'weights_sha256':{'adapter_model.safetensors':sha(out/'adapter_model.safetensors')},'accounting':stats,'same_initial_model':cfg['model_name_or_path'],'fresh_adapter':True,'fresh_optimizer':True}
    if a.smoke:
        gc.collect();torch.cuda.empty_cache()
        base=AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(base,str(out),is_trainable=False).to('cuda').eval()
        loaded=get_peft_model_state_dict(model)
        assert loaded.keys()==weights.keys()
        assert all(torch.equal(v.detach().cpu().to(weights[k].dtype),weights[k]) for k,v in loaded.items())
        from datasets import load_from_disk
        row=load_from_disk(cfg['tokenized_path'])['train'][0]
        ids=torch.tensor([row['input_ids'][:128]],device='cuda')
        with torch.inference_mode():logits=model(input_ids=ids).logits
        assert torch.isfinite(logits).all().item()
        receipt['reload_verification']={'exact_adapter_tensors':len(weights),'real_input_tokens':ids.numel(),'finite_logits':True}
        receipt['status']='SMOKE_TRAIN_SAVE_RELOAD_PASS'
    (out/('SMOKE_DONE.json' if a.smoke else 'TRAINING_DONE.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'output':str(out),'steps':stats['step'],'samples':stats['global_counts']['samples'],'peak_gib':stats['max_memory_allocated']/2**30}),flush=True)
if __name__=='__main__':main()
