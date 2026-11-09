"""Read-only scientific progress snapshot; does not manage processes or score selection."""
from pathlib import Path
import datetime,json
R=Path('/path/to/project');Q=R/'research/function_aware_formal_20261109';L=Path('/path/to/logs/qwen3-multilingual/function-aware-formal-20261109');K=Path('/path/to/storage/checkpoints/function_aware_formal_20261109')
def main():
 jobs=json.loads((R/'configs/function_aware_formal_20261109/jobs.json').read_text());rows=[]
 for name,job in jobs.items():
  directory=K/(name+'_final_v1');marker=directory/'TRAINING_DONE.json';state=L/name/'state.json';v=json.loads(state.read_text()) if state.exists() else {}
  checkpoints=sorted(int(p.name.split('-')[-1]) for p in directory.glob('checkpoint-*'))
  verified=False
  if marker.exists():
   d=json.loads(marker.read_text());assert d['steps']==400 and d['exposure']['samples']==6400 and d['target_order_matches_original_all6400'];verified=True
  rows.append(dict(name=name,language=job['language'],method=job['method'],state='VERIFIED_FINAL400' if verified else v.get('status','WAITING_FIXED_CALIBRATION_WEIGHTS'),phase=v.get('stage','not_started'),latest_saved_checkpoint=max(checkpoints) if checkpoints else None,completed=verified))
 out=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),current_objective='Corrected latest three-language five-method fixed-Final400 plan; supersedes conditional P3 goal and canceled previous crosslayer/SST family; independent FFT supplement separately authorized',new_runs=len(rows),completed=sum(x['completed'] for x in rows),runs=rows,common72_reference='COMPLETE: old36 overlapping sites exactly identical',sw_weights='FROZEN',ar_weights='FROZEN' if (R/'configs/function_aware_formal_20261109/ar_weights.json').exists() else 'PENDING_BUILD_ONLY_CALIBRATION',ru_weights='FROZEN' if (R/'configs/function_aware_formal_20261109/ru_weights.json').exists() else 'PENDING_BUILD_ONLY_CALIBRATION',all_new_formal_results='NOT_AVAILABLE_YET' if not any(x['completed'] for x in rows) else 'CHECK_EVALUATION_DONE_MARKERS',wrong_previous_family='STOPPED_BY_USER; preserve partial KL checkpoint50, teacher cache and unused SST bases; no restart')
 (Q/'progress.json').write_text(json.dumps(out,indent=2)+'\n')
 text=['# Function-Aware正式实验进度','',out['updated_at'],'','当前目标：SW/AR/RU五方法同预算Final400正式比较。旧条件式P3目标与误转跨层求解方案均已被本目标替代。','',f'三语言主线新增正式训练完成：{out["completed"]}/{len(rows)}；历史SW batch2普通LoRA复用。','', '| 任务 | 状态 | 阶段 | 最近已保存节点 |','|---|---|---|---|']
 text += ['| '+x['name']+' | '+x['state']+' | '+x['phase']+' | '+str(x['latest_saved_checkpoint'])+' |' for x in rows]
 text += ['','另有独立FFT补充2组完整训练；它们不计入上述14组，进度见[FFT轨道](../fft_representation_20261109/README.md)。主线和FFT合计16组新增完整训练。','','队列等待或进程存活不代表GPU正在计算；实际GPU归属另由助手核验。未获得的分数不填为0。','', '共同72个特征点已完成，和旧36层末端重合的部分逐值误差为0。CPU检查确认六组映射、均匀损失及梯度正确。','', 'AR/RU官方Global-MMLU材料已准备：各14042题、57学科、5-shot，与SW同一官方revision。','', '本轮不新增独立短训筛选；两步smoke只查代码和曝光计数，不读取benchmark效果。正式测评不会反馈修改规则。']
 (Q/'progress.md').write_text('\n'.join(text)+'\n');print(out['completed'],len(rows))
if __name__=='__main__':main()
