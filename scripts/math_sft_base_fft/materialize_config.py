"""Resolve only the declared storage/runtime profile placeholders in a config."""
import argparse,re
from pathlib import Path
import yaml
from profile_paths import resolve
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists(),a.output
raw=a.input.read_text();text=re.sub(r'\$\{([A-Z_]+)\}',lambda m:resolve(m.group(1)),raw)
before=yaml.safe_load(raw);after=yaml.safe_load(text)
for key,value in before.items():
 if not isinstance(value,str):assert after[key]==value,(key,'non-path configuration changed')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
print(a.output)
