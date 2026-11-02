import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/bench_v2'))
from audit_protocol import verify_rows,exact_overlap,crosssum_overlap
from common import ids_hash
from evalplus_safety import require_execution_proof,validate_container_policy,docker_info_accessible

class DatasetAudit(unittest.TestCase):
 def test_wrong_split_or_direction_rejected(self):
  rows=[{'id':'1','split':'test','direction':'sw->sw'}];cfg={'samples':1,'sample_ids_sha256':ids_hash(rows),'dataset_split':'test','language':'sw->sw'};self.assertTrue(verify_rows(rows,cfg));rows[0]['split']='train'
  with self.assertRaises(ValueError):verify_rows(rows,cfg)
  rows[0]['split']='test';rows[0]['direction']='en->en'
  with self.assertRaises(ValueError):verify_rows(rows,cfg)
 def test_exact_contamination_detects_normalized_fields(self):
  hits=exact_overlap([('train1','Hello WORLD')],[{'id':'test1','source':'hello   world','reference':'other text'}]);self.assertEqual(len(hits),1);self.assertEqual(hits[0]['training_ids'],['train1'])
 def test_crosssum_shared_evidence_not_independent(self):
  cross=[{'source_url':'a','target_url':'b','source':'Text','reference':'Answer'}];source=[{'url':'a','source':'text'}];target=[{'url':'b','reference':'answer'}];r=crosssum_overlap(cross,source,target);self.assertEqual(r['source_url_shared_with_xlsum_test'],1);self.assertEqual(r['exact_target_summary_shared_with_xlsum_test'],1);self.assertFalse(r['semantic_alignment_verified'])

class ExecutionSafety(unittest.TestCase):
 def test_docker_zero_exit_permission_error_is_not_access(self):
  self.assertFalse(docker_info_accessible(0,'null','permission denied'))
  self.assertFalse(docker_info_accessible(0,'null',''))
  self.assertFalse(docker_info_accessible(1,'[]',''))
  self.assertTrue(docker_info_accessible(0,'["name=seccomp"]',''))
 def test_venv_timeout_is_rejected(self):
  for executor in ['venv','timeout','subprocess','bubblewrap_binary_exists']:
   with self.assertRaises(ValueError):require_execution_proof({'executor':executor})
 def test_missing_or_fake_resource_proof_rejected(self):
  with self.assertRaises(ValueError):require_execution_proof({'executor':'verified_container','host_files_inaccessible':True,'network_block_test_passed':True,'resource_enforcement_test_passed':False})
 def test_docker_policy_cannot_have_network_privilege_or_unbounded_limits(self):
  policy=dict(network='none',read_only_root=True,privileged=False,host_pid=False,host_ipc=False,no_new_privileges=True,cap_drop_all=True,devices=[],sensitive_host_mounts=[],limits_measured_in_kernel=True,memory_bytes=2147483648,pids_limit=64,cpus=2,per_task_wall_seconds=60,uid=1026,image_digest='sha256:'+'a'*64)
  self.assertTrue(validate_container_policy(policy))
  for k,v in [('network','host'),('privileged',True),('pids_limit',0),('memory_bytes',None),('limits_measured_in_kernel',False),('uid',0),('devices',['/dev/nvidia0']),('image_digest','python:latest')]:
   bad=dict(policy);bad[k]=v
   with self.assertRaises(ValueError):validate_container_policy(bad)
if __name__=='__main__':unittest.main()
