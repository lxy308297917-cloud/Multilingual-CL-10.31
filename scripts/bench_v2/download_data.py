"""Fetch pinned public archives/JSON data; never execute dataset scripts."""
import argparse,gzip,hashlib,json,os,urllib.request
from pathlib import Path
from common import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for entry in json.loads(a.lock.read_text()):
  name=entry['filename']
  if Path(name).name!=name:raise ValueError('Lock must use simple filenames')
  destination=a.output/name
  if destination.exists():
   if sha256(destination)!=entry['sha256']:raise ValueError('Existing file checksum differs: '+str(destination))
   print('VERIFIED '+name);continue
  if a.verify_only:raise FileNotFoundError(destination)
  if not entry['url'].startswith('https://'):raise ValueError('HTTPS required')
  temporary=destination.with_name(name+'.part')
  with urllib.request.urlopen(entry['url'],timeout=60) as response:
   stream=gzip.GzipFile(fileobj=response) if entry.get('compression')=='gzip' else response
   total=0
   with temporary.open('xb') as output:
    while True:
     chunk=stream.read(1024*1024)
     if not chunk:break
     total+=len(chunk)
     if total>1024**3:raise ValueError('Archive unexpectedly exceeds 1GiB')
     output.write(chunk)
  if sha256(temporary)!=entry['sha256']:raise ValueError('Downloaded checksum differs; .part retained')
  os.rename(temporary,destination);print('DOWNLOADED_VERIFIED '+name)
if __name__=='__main__':main()
