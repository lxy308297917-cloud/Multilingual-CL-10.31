"""Launch the existing LLaMA-Factory workflow with the verified SST adapter."""
import argparse,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/sft_v1'))
from train_formal import ExposureAccounting
from training_adapter import SSTCallback

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--stop-at',type=int);p.add_argument('--audit-moments',action='store_true');a=p.parse_args()
 config=yaml.safe_load(a.config.read_text());sst=config.pop('sst');resume=a.resume or config.get('resume_from_checkpoint')
 if resume:config['resume_from_checkpoint']=str(resume)
 assert config['finetuning_type']=='full' and config['stage']=='sft' and config['weight_decay']==0
 assert config['seed']==42 and config['data_seed']==42 and config['learning_rate']==1e-5
 assert config['train_on_prompt']==False and config['packing']==False
 assert sst['ratio'] in [0,.99] and sst['refresh_interval'] in [None,2,10]
 ds=json.loads(Path(config['deepspeed']).read_text());assert ds['zero_optimization']['stage']==3
 assert 'offload_optimizer' not in ds['zero_optimization']
 from llamafactory.train.tuner import run_exp
 run_exp(config,callbacks=[ExposureAccounting(resume,a.stop_at),SSTCallback(sst['ratio'],sst['refresh_interval'],resume,sst.get('svd_driver','gesvd'),a.audit_moments,sst.get('initial_basis_cache'),sst.get('initial_basis_cache_sha256'))])
if __name__=='__main__':main()
