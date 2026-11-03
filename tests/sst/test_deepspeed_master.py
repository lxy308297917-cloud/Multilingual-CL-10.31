"""Two-rank real ZeRO3 test; must run only after GPU ownership preflight."""
import argparse,json,sys,types
from pathlib import Path
import torch,torch.distributed as dist
import deepspeed
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/sst'))
from training_adapter import SSTCallback
from optimizer_adapter import DeepSpeedMaster

class Network(torch.nn.Module):
 def __init__(self):
  super().__init__();self.config=types.SimpleNamespace(num_hidden_layers=1);self.layer=torch.nn.Module()
  for name,inp,out in [('q_proj',8,8),('k_proj',8,4),('v_proj',8,4),('o_proj',8,8),('gate_proj',8,16),('up_proj',8,16),('down_proj',16,8)]:setattr(self.layer,name,torch.nn.Linear(inp,out,bias=False))
 def forward(self,x):
  a=self.layer.q_proj(x);b=torch.cat([self.layer.k_proj(a),self.layer.v_proj(a)],-1);x=self.layer.o_proj(b);return self.layer.down_proj(torch.nn.functional.silu(self.layer.gate_proj(x))*self.layer.up_proj(x))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();deepspeed.init_distributed();torch.cuda.set_device(int(__import__('os').environ['LOCAL_RANK']));torch.manual_seed(42)
 model=Network().to(torch.bfloat16);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
 config={'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':2,'train_batch_size':4,'gradient_clipping':1.,'bf16':{'enabled':True},'zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True,'stage3_param_persistence_threshold':0,'stage3_gather_16bit_weights_on_model_save':True},'steps_per_print':1000,'zero_allow_untested_optimizer':True}
 engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,config=config);sst=SSTCallback(.99,2,audit_moments=True);sst.on_train_begin(types.SimpleNamespace(output_dir=str(a.output)),types.SimpleNamespace(global_step=0),None,model=engine.module,optimizer=engine.optimizer)
 for micro in range(8):
  x=torch.randn(1,8,device=engine.device,dtype=torch.bfloat16);loss=engine(x).float().square().mean();engine.backward(loss);engine.step();assert sst.step==(micro+1)//2,('Microsteps counted as real steps',micro,sst.step)
  for name,accessor in sst.targets.items():
   current=accessor.get();p=accessor.p;rank=dist.get_rank();start=rank*p.ds_tensor.numel();local=current.flatten()[start:start+p.ds_tensor.numel()];assert torch.equal(p.ds_tensor,local.to(p.ds_tensor.dtype)),'LP/master out of sync: '+name
 assert sst.refresh_steps==[2,4];sst.on_train_end(types.SimpleNamespace(),types.SimpleNamespace(global_step=4),None)
 if dist.get_rank()==0:(a.output/'PASSED.json').write_text(json.dumps({'passed':True,'stage':3,'world_size':2,'microsteps':8,'optimizer_steps':4,'refresh_steps':[2,4],'all_seven_matrices':True,'all_master_writebacks_exact':True,'all_adam_moments_unchanged':True,'all_bf16_partitions_synchronized':True},indent=2))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
