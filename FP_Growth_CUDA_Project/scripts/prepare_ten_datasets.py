#!/usr/bin/env python3
"""Generate six reproducible synthetic baskets and two additional UCI-origin
scikit-learn datasets. Existing UCI WDBC, UCI Digits, and synthetic retail are
re-created if necessary by prepare_datasets.py. Neither labels nor identifiers
are used in transactional features. Always disclose discretization.
"""
import csv
import json
import hashlib
import random
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
SCRIPTS=ROOT/'scripts'

# EXACTLY ten distinct offline data files in the DEFAULT benchmark catalog.
CATALOG=[
    dict(id='uci_iris', label='UCI Iris (3-bin)', filename='uci_iris_150.txt', support='25%', source='UCI-derived (scikit-learn load_iris)', notes='150 samples; four continuous features quantile-discretized into three equal-frequency categories; target omitted'),
    dict(id='uci_wine', label='UCI Wine (median-bin)', filename='uci_wine_178.txt', support='40%', source='UCI-derived (scikit-learn load_wine)', notes='178 samples; 13 real-valued features median-binarized; target omitted'),
    dict(id='uci_wdbc', label='UCI WDBC (median-bin)', filename='uci_wdbc_569.txt', support='40%', source='UCI-derived (scikit-learn load_breast_cancer)', notes='569 samples; 30 features median-binarized; target omitted'),
    dict(id='uci_digits', label='UCI Digits (nonzero)', filename='uci_digits_1797.txt', support='80%', source='UCI-origin handwritten digits test set (scikit-learn load_digits)', notes='1797 samples; nonzero pixels become basket items; digit target omitted'),
    dict(id='syn_grocery', label='Synthetic grocery', filename='synthetic_grocery_3000.txt', support='12%', source='Synthetic (seed 20261029 + 101)', notes='3000 grocery baskets with repeated common staple items and occasional thematic group'),
    dict(id='syn_bundles', label='Synthetic correlated bundles', filename='synthetic_bundles_6000.txt', support='15%', source='Synthetic (seed 20261029 + 202)', notes='6000 baskets with planted clusters and independent background items'),
    dict(id='syn_sparse', label='Synthetic sparse clickstream', filename='synthetic_sparse_12000.txt', support='2%', source='Synthetic (seed 20261029 + 303)', notes='12000 baskets with 2-6 selected items from a 300-item long-tail vocabulary'),
    dict(id='syn_iot', label='Synthetic dense sensors', filename='synthetic_dense_sensors_8000.txt', support='36%', source='Synthetic (seed 20261029 + 404)', notes='8000 simulated dense sensor-state baskets with 32 binary items'),
    dict(id='syn_longtail', label='Synthetic long-tail catalog', filename='synthetic_longtail_20000.txt', support='5%', source='Synthetic (seed 20261029 + 505)', notes='20000 baskets with highly skewed frequency distribution over 150 catalog entries'),
    dict(id='syn_retail', label='Synthetic retail 50k', filename='synthetic_retail_50000.txt', support='5%', source='Synthetic (seed 3582026)', notes='50000 reproducible skewed retail baskets; same set as the previous release'),
]

def dump(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(' '.join(sorted(set(row)))+'\n')


def quantile_bin(sample, q1, q2):
    return 'low' if sample < q1 else ('mid' if sample < q2 else 'high')


def uci_datasets():
    from sklearn.datasets import load_iris, load_wine
    import numpy as np
    iris=load_iris()
    q=np.quantile(iris.data, [1/3,2/3], axis=0)
    dump(DATA/'uci_iris_150.txt',
        ([f'feat{j:02d}='+quantile_bin(v,q[0,j],q[1,j]) for j,v in enumerate(row)]
         for row in iris.data))
    wine=load_wine()
    q=np.median(wine.data,axis=0)
    dump(DATA/'uci_wine_178.txt',
        ([f'feat{j:02d}='+('high' if val>q[j] else 'low') for j,val in enumerate(row)]
         for row in wine.data))


def create_synthetic():
    rng=random.Random(20261029+101)
    vocab=[f'grocery_{i:03d}' for i in range(42)]
    rows=[]
    for _ in range(3000):
        row=rng.sample(vocab[:10], rng.randint(2,5))
        row+=rng.sample(vocab[10:],rng.randint(2,7))
        if rng.random()<.28:row+=vocab[10:13]
        rows.append(row)
    dump(DATA/'synthetic_grocery_3000.txt',rows)

    rng=random.Random(20261029+202)
    clusters=[[f'B{k:02d}_{j:02d}' for j in range(8)] for k in range(5)]
    extra=[f'background_{j:02d}' for j in range(20)]
    rows=[]
    for _ in range(6000):
        k=rng.randrange(5)
        row=rng.sample(clusters[k],rng.randint(4,7))
        if rng.random()<.4: row+=rng.sample(clusters[(k+1)%5],rng.randint(1,3))
        row+=rng.sample(extra,rng.randint(2,6))
        rows.append(row)
    dump(DATA/'synthetic_bundles_6000.txt', rows)

    rng=random.Random(20261029+303)
    vocab=[f'page_{j:03d}' for j in range(300)]
    weights=[1/(j+7)**.75 for j in range(300)]
    rows=[]
    for _ in range(12000):
        rows.append(rng.choices(vocab, weights=weights,k=rng.randint(2,6)))
    dump(DATA/'synthetic_sparse_12000.txt',rows)

    rng=random.Random(20261029+404)
    rows=[]
    for _ in range(8000):
        latent=[rng.random()<.55 for _ in range(4)]
        row=[]
        for j in range(32):
            p=(.66 if latent[j//8] else .23)
            if rng.random()<p:row.append(f'sensor{j:02d}=on')
        if not row:row=['sensor00=on']
        rows.append(row)
    dump(DATA/'synthetic_dense_sensors_8000.txt',rows)

    rng=random.Random(20261029+505)
    vocab=[f'catalog_{j:03d}' for j in range(150)]
    weights=[1/(j+2)**1.02 for j in range(150)]
    rows=[]
    for _ in range(20000):
        rows.append(rng.choices(vocab, weights=weights,k=rng.randint(5,12)))
    dump(DATA/'synthetic_longtail_20000.txt',rows)


def main():
    DATA.mkdir(exist_ok=True)
    # Ensure existing datasets can be recreated; all official four UCI are
    # bundled in the deliverable; this regeneration is optional on lab PCs.
    prereq=['uci_wdbc_569.txt','uci_digits_1797.txt','synthetic_retail_50000.txt']
    if any(not (DATA/n).exists() for n in prereq):
        subprocess.run([sys.executable,str(SCRIPTS/'prepare_datasets.py')], check=True)
    uci_datasets(); create_synthetic()
    for row in CATALOG:
        path=DATA/row['filename']
        if not path.is_file():raise RuntimeError('Missing '+str(path))
        with path.open('r',encoding='utf-8') as f:
            total=0; minlen=10**10; maxlen=0
            for line in f:
                items=line.strip().split()
                if not items:continue
                if len(items)!=len(set(items)): raise AssertionError(f'Duplicate item on row {total}: {path}')
                total+=1;minlen=min(minlen,len(items));maxlen=max(maxlen,len(items))
        row['transactions']=total
        row['min_items_per_row']=minlen
        row['max_items_per_row']=maxlen
        row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{row['id']:<15} {total:>6} transactions  {path.stat().st_size/1024**2:>5.2f} MiB")
    (SCRIPTS/'benchmark_catalog.json').write_text(json.dumps(CATALOG,indent=2)+'\n')
    (DATA/'TEN_DATASET_SOURCES.md').write_text('''# Exactly 10 offline benchmark datasets\n\nAll files here are transaction baskets (one whitespace-separated itemset per line), with **no class target labels**. The ten default benchmarks below include **four UCI-origin datasets and six seeded synthetic datasets**. Synthetic data is not represented as UCI-origin. Files with other names (toy/repeated digits) are supplied for correctness/scaling but are not in the ten-benchmark catalog.\n\n'''+ '\n'.join(f"{i+1}. **{d['label']}** — `{d['filename']}`; {d['transactions']:,} transactions; {d['support']} minimum support; {d['source']}. {d['notes']}" for i,d in enumerate(CATALOG))+'''\n\n## Original dataset references\n\n- Iris (UCI #53): https://archive.ics.uci.edu/dataset/53/iris\n- Wine (UCI #109): https://archive.ics.uci.edu/dataset/109/wine\n- Breast Cancer Wisconsin Diagnostic (UCI #17): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic\n- Optical Recognition of Handwritten Digits (UCI #80): https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits\n- Offline scikit-learn copies: https://scikit-learn.org/stable/datasets/toy_dataset.html\n\n**Preprocessing matters:** Iris uses feature-wise terciles; Wine and WDBC use per-feature median bins; Digits maps positive pixel intensities to pixel-present tokens. The feature transformation and omission of labels change the itemset interpretation.\n\nThe three optional extra original UCI categorical datasets Mushroom, Chess, Connect-4 can be downloaded using `python3 scripts/get_uci.py all` on a network-enabled computer, but they are **not part of the ten offline benchmark inputs** and have **not been benchmarked**.\n''')
    print('All 10 benchmark dataset files verified. Catalog saved:',SCRIPTS/'benchmark_catalog.json')

if __name__=='__main__': main()
