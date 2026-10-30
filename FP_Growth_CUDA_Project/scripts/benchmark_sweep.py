#!/usr/bin/env python3
"""Ten-dataset support-threshold sweeps; measured CPU/CUDA comparisons only.

This is independently written for this project's original C++/CUDA hybrid CLI.
For each dataset, runs 4 decreasing support thresholds with repeated timings;
compares exact itemset counts/max pattern lengths from the two binaries.
Run verify_cuda.py separately to check itemsets and supports individually.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CATALOG=json.loads((ROOT/'scripts/benchmark_catalog.json').read_text())
# Support levels were chosen for these particular transaction encodings to keep
# the experiment tractable and demonstrate a range of frequent pattern counts.
SWEEPS={
 'uci_iris_150.txt':['35%','30%','25%','20%'],
 'uci_wine_178.txt':['50%','45%','40%','30%'],
 'uci_wdbc_569.txt':['50%','45%','40%','35%'],
 'uci_digits_1797.txt':['90%','85%','80%','75%'],
 'synthetic_grocery_3000.txt':['20%','15%','12%','10%'],
 'synthetic_bundles_6000.txt':['20%','17%','15%','12%'],
 'synthetic_sparse_12000.txt':['4%','3%','2%','1.5%'],
 'synthetic_dense_sensors_8000.txt':['45%','40%','36%','30%'],
 'synthetic_longtail_20000.txt':['10%','7.5%','5%','4%'],
 'synthetic_retail_50000.txt':['10%','7.5%','5%','4%'],
}
assert len(CATALOG)==len(SWEEPS)==10
assert set(x['filename'] for x in CATALOG)==set(SWEEPS)
FIELDS=['dataset','filename','origin','min_support','min_support_percent','min_support_count',
 'transactions','unique_items','frequent_itemsets','max_pattern_length','root_tree_nodes',
 'conditional_trees_built','cpu_median_ms','cpu_peak_ram_mib','cuda_median_ms',
 'cuda_peak_ram_mib','cuda_buffer_mib','cuda_kernels','cuda_conditional_kernels',
 'speedup_cpu_over_cuda','gpu_device','runs','verification']

def benchmark(exe, source, minsup, timeout):
    proc=subprocess.run([str(exe),str(source),minsup,'--no-output','--summary','-','--top','0'],
                         capture_output=True,text=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'{exe.name} failed for {source.name} {minsup}: {proc.stderr[-1200:]}')
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f'{exe.name} returned invalid JSON: {proc.stdout[:800]}') from e

def med(s,key,scale=1.):return statistics.median(float(x[key]) for x in s)*scale

def write(out,records,meta):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'support_sweep_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(records)
    (out/'support_sweep_results.json').write_text(json.dumps({**meta,'measurements':records},indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=int,default=240)
    parser.add_argument('--with-cuda',action='store_true',help='Require and measure actual NVIDIA GPU binary')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--datasets',nargs='*',help='Optional filenames or indices (1-10) for trial runs')
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be >=1')
    cpu=ROOT/'fp_growth';gpu=ROOT/'fp_growth_cuda'
    if not cpu.exists():parser.error('Missing CPU binary. Run make')
    if args.with_cuda and not gpu.exists():parser.error('Missing CUDA binary. Run make cuda')
    metadata={'generated_utc':datetime.now(timezone.utc).isoformat(),
              'platform':platform.platform(), 'python':platform.python_version(),
              'cpu_model':platform.processor() or 'unknown',
              'mode':'CPU plus real CUDA' if args.with_cuda else 'CPU only',
              'repeats':args.repeats,'notes':'Times include input parsing plus mining, exclude Python subprocess startup, and exclude CSV output when using --no-output. Each row is median over independent processes. Peak RAM is process RSS.'}
    if Path('/proc/cpuinfo').exists():
        metadata['cpu_model']=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),metadata['cpu_model'])
    rec=[]
    for index,ds in enumerate(CATALOG,1):
        f=ds['filename']
        if args.datasets and f not in args.datasets and str(index) not in args.datasets:continue
        source=ROOT/'data'/f
        if not source.exists():raise SystemExit(f'Missing dataset {source}')
        for sup in SWEEPS[f]:
            c_runs=[];g_runs=[]
            for repeat in range(args.repeats):
                # Alternate execution order (CPU/GPU) to reduce systematic temperature and caching bias.
                if args.with_cuda and repeat%2:
                    gr=benchmark(gpu,source,sup,args.timeout)
                    cr=benchmark(cpu,source,sup,args.timeout)
                else:
                    cr=benchmark(cpu,source,sup,args.timeout)
                    gr=benchmark(gpu,source,sup,args.timeout) if args.with_cuda else None
                c_runs.append(cr)
                if gr:
                    if cr['frequent_itemsets']!=gr['frequent_itemsets'] or cr['max_pattern_length']!=gr['max_pattern_length']:
                        raise RuntimeError(f'CPU/CUDA disagreement on {f} {sup}. Run strict verify_cuda.py')
                    if gr.get('backend')!='cuda-hybrid-all-level-counting' or 'SIMULATOR' in gr.get('cuda_device','').upper():
                        raise RuntimeError(f'CUDA binary is not confirmed as real NVIDIA GPU for {f}')
                    g_runs.append(gr)
            row={'dataset':ds['label'],'filename':f,'origin':('UCI-derived' if f.startswith('uci_') else 'Synthetic'),
                 'min_support':sup,'min_support_percent':float(sup.rstrip('%')),
                 'min_support_count':cr['minimum_support_count'],
                 'transactions':cr['transactions'],'unique_items':cr['unique_items'],
                 'frequent_itemsets':cr['frequent_itemsets'],'max_pattern_length':cr['max_pattern_length'],
                 'root_tree_nodes':cr['root_tree_nodes'],'conditional_trees_built':cr['conditional_trees_built'],
                 'cpu_median_ms':round(med(c_runs,'runtime_seconds',1000),4),
                 'cpu_peak_ram_mib':round(med(c_runs,'peak_rss_kib',1/1024),3),
                 'cuda_median_ms':'','cuda_peak_ram_mib':'','cuda_buffer_mib':'',
                 'cuda_kernels':'','cuda_conditional_kernels':'','speedup_cpu_over_cuda':'',
                 'gpu_device':'','runs':args.repeats,'verification':'CPU counts consistent'}
            if g_runs:
                cpu_t=med(c_runs,'runtime_seconds',1000)
                cuda_t=med(g_runs,'runtime_seconds',1000)
                row.update(cuda_median_ms=round(cuda_t,4),
                           cuda_peak_ram_mib=round(med(g_runs,'peak_rss_kib',1/1024),3),
                           cuda_buffer_mib=round(max(float(x['cuda_buffer_high_water_bytes']) for x in g_runs)/1048576,3),
                           cuda_kernels=g_runs[0]['cuda_kernel_calls'],
                           cuda_conditional_kernels=g_runs[0]['support_conditional_calls'],
                           speedup_cpu_over_cuda=round(cpu_t/cuda_t,4) if cuda_t else '',
                           gpu_device=g_runs[0]['cuda_device'],
                           verification='CPU/GPU pattern count & max length match; run strict itemset verifier')
            rec.append(row)
            write(args.out,rec,metadata)  # incremental writes survive an interrupted lab session
            print(f'[{index:02d}/10] {f:<34} {sup:>5}  {row["frequent_itemsets"]:>6} itemsets  CPU {row["cpu_median_ms"]:>9.3f} ms'+
                  (f'  GPU {row["cuda_median_ms"]:>9.3f} ms  {row["speedup_cpu_over_cuda"]:.3f}x' if g_runs else ''),flush=True)
    print(f'Saved {len(rec)} measured support settings to {args.out/"support_sweep_results.csv"}')
    if not args.datasets and len(rec)!=40:raise SystemExit(f'Expected 40 rows, got {len(rec)}')

if __name__=='__main__':main()
