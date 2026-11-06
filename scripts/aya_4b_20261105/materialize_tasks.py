"""Expand three declared data/runtime prefixes; preserve Harness YAML tags."""
import argparse,os,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--belebele-data',type=Path,required=True);p.add_argument('--benchmark-cache',type=Path,required=True);a=p.parse_args();assert not a.output.exists();spec=importlib.util.find_spec('lm_eval');assert spec and spec.submodule_search_locations;harness=Path(next(iter(spec.submodule_search_locations)))/'tasks';values={'BELEBELE_DATA':a.belebele_data,'BENCHMARK_CACHE':a.benchmark_cache,'HARNESS_TASKS':harness}
for source in a.source.rglob('*'):
 if source.is_file() and source.suffix in ['.yaml','.py']:
  text=source.read_text()
  for key,value in values.items():text=text.replace('${'+key+'}',str(value.resolve()))
  assert '${' not in text;target=a.output/source.relative_to(a.source);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
