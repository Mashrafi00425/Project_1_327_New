#!/usr/bin/env python3
"""Re-run reproducible FP-Growth experiments; writes CSV + JSON reports."""
import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'fp_growth'
RESULTS = ROOT / 'results'
CATALOG = json.loads((ROOT / 'scripts' / 'benchmark_catalog.json').read_text())
CONFIG = [(x['label'], x['filename'], x['support']) for x in CATALOG]
assert len(CONFIG) == 10 and len({x[1] for x in CONFIG}) == 10, 'Expected exactly ten distinct benchmarks'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not EXE.exists():
        subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',
                        str(ROOT/'src/fp_growth.cpp'),'-o',str(EXE)], check=True)
    records = []
    for dataset, name, support in CONFIG:
        filename = ROOT / 'data' / name
        if not filename.exists():
            raise FileNotFoundError(f'Missing required benchmark dataset: {filename}. Run python3 scripts/prepare_ten_datasets.py')
        runs = []
        for i in range(args.repeats):
            proc = subprocess.run([str(EXE),str(filename),support,
                                   '--no-output','--summary','-', '--top','5'],
                                  capture_output=True,text=True, timeout=90,check=True)
            runs.append(json.loads(proc.stdout))
        base = runs[0]
        assert all(x['frequent_itemsets'] == base['frequent_itemsets'] for x in runs)
        rec = dict(dataset=dataset,file=name,transactions=base['transactions'],
                   unique_items=base['unique_items'],support=support,
                   support_count=base['minimum_support_count'],
                   patterns=base['frequent_itemsets'],max_length=base['max_pattern_length'],
                   root_tree_nodes=base['root_tree_nodes'],
                   runtime_median_s=round(statistics.median(r['runtime_seconds'] for r in runs),6),
                   runtime_min_s=round(min(r['runtime_seconds'] for r in runs),6),
                   runtime_max_s=round(max(r['runtime_seconds'] for r in runs),6),
                   peak_rss_median_mib=round(statistics.median(r['peak_rss_kib'] for r in runs)/1024,3),
                   peak_rss_max_mib=round(max(r['peak_rss_kib'] for r in runs)/1024,3))
        records.append(rec)
        print(f'{dataset:<30} {support:<4} | {rec["patterns"]:>7} patterns | {rec["runtime_median_s"]:8.4f} s | {rec["peak_rss_median_mib"]:7.2f} MiB')
    if len(records) != 10:
        raise SystemExit(f'Expected ten datasets, got {len(records)}')
    with (RESULTS/'benchmark_results.csv').open('w',newline='',encoding='utf-8') as fp:
        writer = csv.DictWriter(fp,fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    extra = {
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'compiler': subprocess.run(['g++','--version'],capture_output=True,text=True).stdout.splitlines()[0],
            'cpu_model': next((line.split(':',1)[1].strip() for line in
                               Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                              'unknown') if Path('/proc/cpuinfo').exists() else platform.processor(),
            'timing': f'wall-clock elapsed seconds, median of {args.repeats} independent processes',
            'memory': 'Linux /proc/self/status VmHWM peak resident memory KiB; medians in MiB; alternatives use getrusage',
        },
        'experiments': records
    }
    (RESULTS/'benchmark_results.json').write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8')
    print('\nSaved results/benchmark_results.csv and .json')

if __name__ == '__main__':
    main()
