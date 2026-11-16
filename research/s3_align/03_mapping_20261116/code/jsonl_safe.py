"""Read JSONL using physical LF lines, preserving legal Unicode U+2028/U+2029."""
import json
from pathlib import Path
def rows(path):
 with Path(path).open(encoding='utf-8',newline='') as f:
  return [json.loads(line) for line in f if line.strip()]
