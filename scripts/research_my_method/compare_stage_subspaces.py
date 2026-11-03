"""Sign/rotation-invariant overlap of numerically represented SST projectors."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch

def trace_free_product(a,b,dim,afree,bfree):
 inner=float((a.T@b).square().sum())
 if afree and bfree:return inner
 if afree:return a.shape[1]-inner
 if bfree:return b.shape[1]-inner
 return dim-a.shape[1]-b.shape[1]+inner

def independent_dense_gate():
 torch.manual_seed(72)
 for m in [5,8]:
  a=torch.linalg.qr(torch.randn(m,3,dtype=torch.float64)).Q;b=torch.linalg.qr(torch.randn(m,2,dtype=torch.float64)).Q
  for af in [False,True]:
   for bf in [False,True]:
    pa=a@a.T;pb=b@b.T
    if not af:pa=torch.eye(m,dtype=torch.float64)-pa
    if not bf:pb=torch.eye(m,dtype=torch.float64)-pb
    expected=float(torch.trace(pa@pb));actual=trace_free_product(a,b,m,af,bf);assert abs(expected-actual)<1e-10

def main():
 p=argparse.ArgumentParser()
 for k in ['old','new','sst-code','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--label',required=True);p.add_argument('--old-expected-sha',required=True);p.add_argument('--new-expected-sha')
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.time();torch.set_num_threads(4);torch.set_grad_enabled(False);independent_dense_gate();sys.path.insert(0,str(a.sst_code))
 # Inputs must be completed, immutable training artifacts. Expected hashes below
 # are taken from the source trainer's verified receipt; mmap avoids rereading
 # all reference tensor pages merely to recompute a file digest.
 old=torch.load(a.old,map_location='cpu',weights_only=False,mmap=True);new=torch.load(a.new,map_location='cpu',weights_only=False,mmap=True)
 assert len(old['bases'])==len(new['bases'])==196 and set(old['bases'])==set(new['bases'])
 rows=[]
 for name in sorted(old['bases']):
  x,y=old['bases'][name],new['bases'][name];assert x.shape==y.shape and x.rank==y.rank
  m,n=x.shape;dl=m-x.rank;dr=n-x.rank
  l=trace_free_product(x.left.double(),y.left.double(),m,x.left_free,y.left_free);rr=trace_free_product(x.right.double(),y.right.double(),n,x.right_free,y.right_free)
  overlap=l*rr/(dl*dr);assert -1e-3<=overlap<=1+1e-3
  rows.append({'name':name,'free_dimension':dl*dr,'left_free_rank':dl,'right_free_rank':dr,'left_trace_product':l,'right_trace_product':rr,'free_operator_overlap_fraction_raw':overlap,'interpretation':'Trace(Fold Fnew)/rank(Fold), numerical factors; not per-vector cosine'})
  if len(rows)%28==0:(a.output/'progress.json').write_text(json.dumps({'completed':len(rows),'seconds':time.time()-start})+'\n')
 total=sum(x['free_dimension'] for x in rows);mean=sum(x['free_operator_overlap_fraction_raw']*x['free_dimension'] for x in rows)/total
 report={'status':'PROJECTOR_GEOMETRY_COMPLETE','label':a.label,'old_file':str(a.old),'new_file':str(a.new),'old_expected_file_sha256_from_trainer':a.old_expected_sha,'new_expected_file_sha256_from_trainer':a.new_expected_sha,'file_digests_independently_recomputed_here':False,
  'new_saved_step':new.get('step',0),'matrices':196,'total_free_dimension':total,'dimension_weighted_overlap_fraction_raw':mean,'minimum_matrix_overlap_fraction_raw':min(x['free_operator_overlap_fraction_raw'] for x in rows),'maximum_matrix_overlap_fraction_raw':max(x['free_operator_overlap_fraction_raw'] for x in rows),'rows':rows,'seconds':time.time()-start,'dense_representation_gate_passed':True,
  'limitations':['Numerical projector overlap, not semantic information or causal protection.','Float32 factors are not perfectly orthonormal; raw trace can marginally exceed rank.','Does not independently verify singular-value noncrossing inequality.','Expected file hashes are recorded source identities, not independently rehashed by this mmap analysis.']}
 (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['label','new_saved_step','dimension_weighted_overlap_fraction_raw','minimum_matrix_overlap_fraction_raw','maximum_matrix_overlap_fraction_raw','seconds']}))
if __name__=='__main__':main()
