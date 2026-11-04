"""Fetch the isolated pinned author ROUGE package with exact byte validation."""
import json
from common import ROOT
from fetch_sources import fetch

def main():
    lock = json.loads((ROOT/'configs/unified_language_cl/metrics_sources.lock.json').read_text())
    for asset in lock['assets']:
        fetch(asset)
    print(json.dumps({'status':'VERIFIED','files':len(lock['assets']),'upstream':lock['upstream'],'installed_environments_modified':False}))

if __name__ == '__main__':
    main()
