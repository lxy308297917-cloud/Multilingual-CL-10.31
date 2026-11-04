"""Fetch checksum-locked public data only. Never execute remote dataset loaders."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
import requests
from common import ROOT, DATA
from runtime import runtime_path

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):value.update(part)
    return value.hexdigest()

def valid(path, asset):
    return path.is_file() and path.stat().st_size==asset['bytes'] and sha(path)==asset['sha256']

def fetch(asset):
    destination=Path(runtime_path(asset['root']))/asset['path']
    if destination.exists():
        if not valid(destination,asset):raise ValueError('Existing source differs from lock; preserve it and select a new data namespace: '+asset['path'])
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.download.part')
    url=asset['url'];block=2*1024**2;size=asset['bytes']
    if size<8*1024**2:
        response=requests.get(url,stream=True,timeout=(30,180));response.raise_for_status()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024**2):stream.write(chunk)
    else:
        parts=destination.parent/'.download_parts'/asset['sha256'];parts.mkdir(parents=True,exist_ok=True)
        def piece(start):
            end=min(start+block,size)-1;part=parts/str(start)
            if part.exists() and part.stat().st_size==end-start+1:return part
            for attempt in range(2):
                try:
                    separator='&' if '?' in url else '?';r=requests.get(url+separator+f'download=true&offset={start}&attempt={attempt}',headers={'Range':f'bytes={start}-{end}'},timeout=(30,120));r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Server did not return the requested byte range')
                    if len(r.content)!=end-start+1:raise ValueError('Incomplete byte range')
                    temp=part.with_suffix('.tmp');temp.write_bytes(r.content);temp.replace(part);return part
                except (requests.RequestException,ValueError):
                    if attempt:raise
        # Servers without range support use one normal streaming transfer.
        probe=requests.get(url,headers={'Range':f'bytes=0-{min(block,size)-1}'},stream=True,timeout=(30,120));probe.raise_for_status()
        if probe.status_code==200:
            with temporary.open('wb') as stream:
                for chunk in probe.iter_content(1024**2):stream.write(chunk)
        else:
            probe.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(piece,range(0,size,block)))
            with temporary.open('wb') as stream:
                for start in range(0,size,block):
                    with (parts/str(start)).open('rb') as part:
                        for chunk in iter(lambda:part.read(1024**2),b''):stream.write(chunk)
        if valid(temporary,asset):
            for part in parts.glob('*'):part.unlink()
            parts.rmdir()
    if not valid(temporary,asset):raise ValueError('Downloaded bytes differ from frozen SHA/size: '+asset['path'])
    if destination.exists():raise FileExistsError(destination)
    os.replace(temporary,destination)
    return destination

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock',type=Path,default=ROOT/'configs/unified_language_cl/sources.lock.json');parser.add_argument('--tracks',nargs='+');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();lock=json.loads(args.lock.read_text());selected=[a for a in lock['assets'] if not args.tracks or set(args.tracks)&set(a['tracks'])];assert selected
    receipts=[]
    for asset in selected:
        destination=Path(runtime_path(asset['root']))/asset['path']
        if args.verify_only:
            if not valid(destination,asset):raise ValueError('Missing or mismatched source '+asset['path'])
        else:fetch(asset)
        receipts.append({'root':asset['root'],'path':asset['path'],'bytes':asset['bytes'],'sha256':asset['sha256']});print(json.dumps(receipts[-1]),flush=True)
    if not args.verify_only:
        for repo,metadata in lock['metadata'].items():
            target=DATA/'sources'/repo.replace('/','--')/'metadata.json';target.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(metadata,ensure_ascii=False,indent=2)+'\n'
            if target.exists():
                existing=json.loads(target.read_text());assert existing['sha']==metadata['sha'] and {x['rfilename'] for x in existing.get('siblings',[])}=={x['rfilename'] for x in metadata.get('siblings',[])},'Existing source metadata differs; choose new data namespace '+repo
            if not target.exists():target.write_text(text)
        gem=DATA/'sources/GEM--wiki_lingua/wikilingua_cleaned.tar.gz'
        if any(a['path'].endswith('wikilingua_cleaned.tar.gz') for a in selected):
            receipt=gem.with_name('download_verified.json')
            if not receipt.exists():receipt.write_text(json.dumps({'revision':lock['metadata']['GEM/wiki_lingua']['sha'],'lfs_sha256':sha(gem),'download_sha256_verified':True,'source_acceptance':'Pinned author-supported GEM release; no remote loader code executed.'},indent=2)+'\n')
        if any(a['root']=='belebele_source' for a in selected):
            folder=Path(runtime_path('belebele_source'));target=folder/'eng_Latn.jsonl';expected=lock['derived_belebele_eng_Latn']
            with zipfile.ZipFile(folder/'Belebele.zip') as archive:content=archive.read(expected['member'])
            assert hashlib.sha256(content).hexdigest()==expected['sha256']
            if target.exists():assert target.read_bytes()==content
            else:target.write_bytes(content)
        proof=DATA/'source_verification.json';proof.write_text(json.dumps({'lock_sha256':sha(args.lock),'verified':receipts,'remote_code_executed':False},indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','assets':len(receipts),'remote_code_executed':False}))
if __name__=='__main__':main()
