"""CPU-only serial model scoring, two isolated official EvalPlus workers per model."""
import datetime,json,os,subprocess,time
from pathlib import Path
R=Path('/data0/users/lixiaoyang/10.31/multilingual-research')
L=Path('/data0/users/lixiaoyang/logs/qwen3-multilingual/representation-benchmarks-20261109')
A=Path('/data0/users/lixiaoyang/Documents/Codex/2026-10-30/data0-users-lixiaoyang')
P='/data1/lixiaoyang/envs/qwen3-multilingual-20261031/bin/python'
def main():
    assert (A/'work/representation_benchmarks_20261109/sandbox_preflight_v1/VERIFIED.json').exists()
    deadline=time.monotonic()+24*3600;remaining=['Base','all_layer','full_feature','ce_replay'];calibration=None
    def state(status,**more):(L/'code-execution-state.json').write_text(json.dumps(dict(status=status,remaining=remaining,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**more),indent=2)+'\n')
    while remaining:
        assert time.monotonic()<deadline,'Finite deadline expired; no execution retries'
        candidates=[m for m in remaining if (L/('formal-'+m.lower()+'-code')/'generation/DONE.json').exists()]
        if not candidates:state('WAITING_GENERATION');time.sleep(30);continue
        m=candidates[0];root=L/('formal-'+m.lower()+'-code');out=root/'execution';state('SCORING_ISOLATED_CPU',model=m)
        cmd=[P,str(R/'scripts/math_sft_base_fft/execute_code.py'),'--generation',str(root/'generation'),'--output',str(out),'--role',m]
        if calibration is not None:cmd+=['--calibration',str(calibration)]
        try:
            with (root/'execution.launch.log').open('x') as f:subprocess.run(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
            assert (out/'DONE.json').exists()
            if calibration is None:calibration=out
            remaining.remove(m)
        except Exception as e:state('FAILED',model=m,error=repr(e));raise
    state('ALL_FOUR_REAL_EXECUTION_COMPLETE')
if __name__=='__main__':main()
