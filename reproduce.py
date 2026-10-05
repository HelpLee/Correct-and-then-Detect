"""Released weights -> metrics -> figure files -> immutable reference comparison."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,'-B',str(ROOT/script),*args],cwd=ROOT,check=True)

def verify():
    for manifest in ('source_manifest.json','artifacts/manifest.json'):
        path=ROOT/manifest
        if not path.exists(): raise FileNotFoundError(f'Missing {manifest}: extract the artifact bundle into this repository.')
        for row in json.loads(path.read_text(encoding='utf-8')):
            f=ROOT/row['path']
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError(f'Checksum mismatch: {row["path"]}')
    print('PASS: source/reference and artifact checksums match; generated outputs are excluded.')

def compare():
    checks=[]
    for stage in ('nominal','detection'):
        checks += [c for c in json.loads((ROOT/'results/generated'/f'{stage}_audit.json').read_text(encoding='utf-8'))['checks'] if c['name'] != 'window_sweep']
    for check in checks: print(check['status'],check['name'],check['maximum_absolute_delta'])
    descriptive=ROOT/'results/generated/descriptive_audit.json'
    descriptive_checks=[]
    if descriptive.exists():
        descriptive_checks=json.loads(descriptive.read_text(encoding='utf-8'))
        for row in descriptive_checks: print(row['status'],row['item'])
    return 0 if all(c['status'] in {'PASS','ROUNDING_DIFFERENCE'} for c in checks+descriptive_checks) else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','smoke','nominal','detection','evaluate','figures','compare','full','windows'])
    args=parser.parse_args()
    if args.command=='verify': verify()
    elif args.command=='windows':
        run('experiments/window_sensitivity/verify_results.py')
        run('scripts/plotting/fig10_window_sensitivity.py')
    elif args.command=='smoke': run('scripts/smoke_checkpoints.py')
    else:
        if args.command in {'nominal','evaluate','full'}: run('scripts/evaluate_checkpoints.py','nominal')
        if args.command in {'detection','evaluate','full'}: run('scripts/evaluate_checkpoints.py','detection')
        if args.command in {'figures','full'}:
            run('scripts/render_results.py')
            run('scripts/summarize_audit.py')
        if args.command in {'compare','full'}: sys.exit(compare())

if __name__=='__main__': main()
