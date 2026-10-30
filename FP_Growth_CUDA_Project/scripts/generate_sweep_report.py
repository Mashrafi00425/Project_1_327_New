#!/usr/bin/env python3
"""Generate a genuine multi-support benchmark PDF from support_sweep_results.csv.

Unlike the older one-threshold report, this prints four panels per dataset:
CPU/CUDA time, itemset count, host/device RAM, and speedup versus support.
If lab measurements are absent, it explicitly says GPU data are pending.
No data from a reference/friend's project are imported.
"""
from __future__ import annotations
import csv
import html
import json
import math
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether
except ImportError as exc:
    raise SystemExit(f'Missing PDF dependency: {exc}. On Ubuntu: sudo apt install python3-reportlab python3-matplotlib')

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'/'FP_Growth_Expanded_10_Dataset_Report.pdf'
RESULTS=ROOT/'results'
CATALOG=json.loads((ROOT/'scripts'/'benchmark_catalog.json').read_text())
DATA=RESULTS/'support_sweep_results.csv'
if not DATA.exists():raise SystemExit('First run: python3 scripts/benchmark_sweep.py --repeats 3')
with DATA.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
by_file={x['filename']:[] for x in CATALOG}
for r in rows:
    if r['filename'] in by_file:by_file[r['filename']].append(r)
if len(rows)!=40 or any(len(v)!=4 for v in by_file.values()):
    raise SystemExit(f'Incomplete support sweep: {len(rows)} rows. Expected exactly 40 rows, 4 per dataset. Run scripts/benchmark_sweep.py without --datasets.')
for value in by_file.values():value.sort(key=lambda r:float(r['min_support_percent']),reverse=True)
all_have_gpu=all(r['cuda_median_ms']!='' and r['gpu_device'] and 'SIMULATOR' not in r['gpu_device'].upper() for r in rows)
any_have_gpu=any(r['cuda_median_ms']!='' for r in rows)
if any_have_gpu and not all_have_gpu:
    raise SystemExit('Incomplete GPU measurements in support_sweep_results.csv: refusing mixed-mode comparisons.')
mode='Actual NVIDIA CUDA vs CPU, same-machine' if all_have_gpu else 'CPU measurements; CUDA lab results pending'
META=json.loads((RESULTS/'support_sweep_results.json').read_text())
if len(META.get('measurements',[]))!=40:
    raise SystemExit('CSV/JSON metadata mismatch; re-run benchmark_sweep.py')
META_DEVICE=rows[0]['gpu_device'] if all_have_gpu else None

COLORS={'cpu':'#12668a','gpu':'#ed8a20','memory':'#3a8e62','patterns':'#26344a'}
chart_dir=RESULTS/'sweep_figures';chart_dir.mkdir(parents=True,exist_ok=True)

def asf(r,key):return float(r[key])

def plot_chart(rs,idx):
    name=CATALOG[idx]['label'];x=[asf(r,'min_support_percent') for r in rs]
    fig,axs=plt.subplots(2,2,figsize=(9.0,6.5),dpi=155)
    ax=axs[0,0]
    ax.plot(x,[max(1e-6,asf(r,'cpu_median_ms')) for r in rs],'o-',lw=2,label='CPU',color=COLORS['cpu'])
    if all_have_gpu:
        ax.plot(x,[max(1e-6,asf(r,'cuda_median_ms')) for r in rs],'s-',lw=2,label='CUDA',color=COLORS['gpu'])
    ax.set_yscale('log');ax.set_title('Runtime (parse + mine, ms)');ax.set_ylabel('ms (log scale)');ax.legend(fontsize=8)
    ax=axs[0,1]
    ax.plot(x,[max(1,asf(r,'frequent_itemsets')) for r in rs],'o-',lw=2,color=COLORS['patterns'])
    ax.set_yscale('log');ax.set_title('Number of frequent itemsets');ax.set_ylabel('itemsets (log scale)')
    ax=axs[1,0]
    ax.plot(x,[asf(r,'cpu_peak_ram_mib') for r in rs],'o-',lw=2,label='CPU host RAM',color=COLORS['cpu'])
    if all_have_gpu:
        ax.plot(x,[asf(r,'cuda_peak_ram_mib') for r in rs],'s-',lw=2,label='CUDA host RAM',color=COLORS['gpu'])
        ax.plot(x,[asf(r,'cuda_buffer_mib') for r in rs],'^--',lw=1.5,label='CUDA buffer high-water',color=COLORS['memory'])
    ax.set_title('Peak memory');ax.set_ylabel('MiB');ax.legend(fontsize=8)
    ax=axs[1,1]
    if all_have_gpu:
        ax.plot(x,[asf(r,'speedup_cpu_over_cuda') for r in rs],'o-',color=COLORS['cpu'],lw=2)
        ax.axhline(1,color='#999999',lw=1,ls='--')
        ax.set_title('Speedup = CPU time / CUDA time');ax.set_ylabel('ratio (over 1 means GPU faster)')
    else:
        ax.text(.5,.5,'Pending actual NVIDIA GPU results\nRun benchmark_sweep.py --with-cuda',
                ha='center',va='center',transform=ax.transAxes,fontsize=10,color='#657487')
        ax.set_title('CUDA speedup pending');ax.set_xticks([]);ax.set_yticks([])
    for a in axs.flat:
        if a is axs[1,1] and not all_have_gpu:continue
        a.set_xlabel('Minimum support (%)')
        a.set_xticks(x)
        a.invert_xaxis()
        a.grid(alpha=.18,which='both')
    fig.suptitle(f'{idx+1:02d}  {name}  ({rs[0]["transactions"]} transactions)',fontsize=14,weight='semibold')
    fig.tight_layout(rect=(0,0,1,.95))
    path=chart_dir/f'dataset_{idx+1:02}.png'
    fig.savefig(path,bbox_inches='tight',dpi=150)
    plt.close(fig)
    return path

FIGURES={ds['filename']:plot_chart(by_file[ds['filename']],i) for i,ds in enumerate(CATALOG)}
NAVY=colors.HexColor('#123349');TEAL=colors.HexColor('#087b86');LINE=colors.HexColor('#d5dce1')
PALE=colors.HexColor('#f3f7f9');INK=colors.HexColor('#1d303e')
ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='CoverX',fontName='Helvetica-Bold',fontSize=24,leading=30,textColor=NAVY,spaceAfter=14))
ss.add(ParagraphStyle(name='SectionX',fontName='Helvetica-Bold',fontSize=14,leading=18,textColor=NAVY,spaceBefore=10,spaceAfter=11,keepWithNext=True))
ss.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=10,leading=14,textColor=TEAL,spaceBefore=9,spaceAfter=7,keepWithNext=True))
ss.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.3,leading=14,textColor=INK,spaceAfter=9))
ss.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=7.5,leading=10.5,textColor=INK,spaceAfter=7))
ss.add(ParagraphStyle(name='TableHeadX',fontName='Helvetica-Bold',fontSize=8,leading=10.4,textColor=colors.white))
ss.add(ParagraphStyle(name='TableBodyX',fontName='Helvetica',fontSize=8,leading=10.7,textColor=INK))
ss.add(ParagraphStyle(name='CodeX2',fontName='Courier',fontSize=8,leading=12,backColor=PALE,leftIndent=7,spaceAfter=9))
P=lambda s,style='BodyX':Paragraph(s,ss[style])
H=lambda s:P(s,'SectionX')
S=lambda s:P(s,'SubX')
TXT=lambda s:html.escape(str(s))

def table(lines,widths,fontsize=8):
    data=[[P(TXT(c),'TableHeadX' if i==0 else 'TableBodyX') for c in line] for i,line in enumerate(lines)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
       ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),6),
       ('RIGHTPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,0),(-1,0),NAVY),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LINEBELOW',(0,-1),(-1,-1),.4,LINE)]))
    return t

def footer(cv,doc):
    cv.saveState();w,h=letter
    cv.setStrokeColor(LINE);cv.line(.67*inch,h-.5*inch,w-.67*inch,h-.5*inch)
    cv.line(.67*inch,.51*inch,w-.67*inch,.51*inch)
    cv.setFont('Helvetica',7.5);cv.setFillColor(NAVY)
    cv.drawString(.69*inch,h-.4*inch,'FP-GROWTH / CUDA HYBRID / SUPPORT-THRESHOLD SWEEPS')
    cv.drawString(.69*inch,.37*inch,'10 independent input datasets | CPU & CUDA benchmark methodology')
    cv.drawRightString(w-.69*inch,.37*inch,str(doc.page));cv.restoreState()

story=[]
story += [Spacer(1,.34*inch),P('CUDA-Accelerated FP-Growth<br/>Support-Threshold Study','CoverX'),
 P('Ten datasets, forty support settings, and CPU/CUDA performance analysis','BodyX'),
 Spacer(1,.18*inch),H('Abstract'),
 P('This project implements complete frequent-itemset mining using FP-Growth: a compressed frequent-pattern tree, linked headers, conditional pattern bases and recursive conditional trees. A separate CUDA build accelerates weighted support-counting histograms for the initial database and every recursive conditional tree. Tree construction and recursion remain on the CPU, making this a genuine <b>hybrid CUDA</b> implementation, not a fully GPU-resident tree miner.'),
 P('Experiments evaluate ten distinct transaction datasets at four minimum-support thresholds each. Per-threshold measurements include exact frequent-itemset count, CPU time, host resident memory, and - where actual NVIDIA measurements exist - CUDA time, CUDA memory, kernel counts and CPU/CUDA speedup. Each dataset has its own four-panel performance figure and a results table.'),
 P('<b>Dataset provenance:</b> four UCI-derived datasets and six reproducibly generated synthetic workloads. These are not ten UCI datasets. All measurements are from this project; none were imported from the comparison project.'),
 table([['Measured settings','Benchmark status','GPU status'],['10 x 4 = 40','CPU experiments completed', 'Real GPU runs recorded' if all_have_gpu else 'Real GPU sweep pending']],[155,168,180]),
 Spacer(1,.15*inch),P('<b>Measured environment:</b> '+TXT(META.get('cpu_model','unknown'))+'; '+TXT(META.get('platform','unknown'))+'; '+('GPU: '+TXT(META_DEVICE) if all_have_gpu else 'GPU: not benchmarked in these records')+'.','SmallX'),
 P('<b>Timing:</b> median of '+TXT(META.get('repeats','?'))+' independent runs per support level; C++ process timing including input parsing and mining, excluding Python startup and pattern CSV writing. Very small runs are sensitive to timer noise.','SmallX'),PageBreak()]

story += [H('1. Problem definition and FP-Growth method'),
 P('Given a transaction database D with N rows and an itemset X, support(X) is the number of transactions containing every item of X. At minimum support s, X is frequent if support(X) is at least ceil(sN). Our miner outputs every nonempty frequent itemset and its support, with duplicates removed inside each transaction.'),
 S('1.1 Sequential miner'),
 P('The initial scan counts item supports and discards infrequent tokens. Remaining items are sorted by decreasing support and inserted into an FP-tree. Shared prefixes share nodes. Header links connect nodes with the same item. For each suffix item, node-to-root ancestor paths form a weighted conditional pattern base. A conditional FP-tree is then built and recursively mined.'),
 S('1.2 CUDA hybrid: actual GPU responsibilities'),
 P('For both the root tree and recursively generated conditional pattern bases, CPU code flattens item IDs and path multiplicities into arrays. These arrays are copied to the NVIDIA device. A 64-bit atomicAdd histogram kernel computes weighted support values in parallel. The histogram returns to the CPU, which prunes and rebuilds the tree. This can accelerate large counting workloads but may be slower for many tiny conditional trees because of kernel-launch and transfer overhead.'),
 Preformatted('''CPU: generate weighted transactions from conditional pattern base
  -> flatten [item_id], [weight]
GPU: transfer arrays, reset support histogram, launch atomicAdd kernel
  -> transfer support histogram back to host
CPU: remove infrequent items, build conditional tree, recurse''',ss['CodeX2']),
 S('1.3 Correctness approach'),
 P('The weighted support of item j in a conditional base is the sum of path weights of all rows containing j. Each suffix ordering is explored once, so every frequent itemset is emitted once. The same input and threshold are evaluated with the CPU and actual CUDA executables. Their complete itemset/support CSVs are compared by scripts/verify_cuda.py, alongside small independent exhaustive checks. Benchmark summary equality alone is not a substitute for full pattern verification.'),
 PageBreak()]

story += [H('2. Manual validation and reproducibility'),
 P('The five-basket toy example contains bread, milk, diaper, beer and eggs. With absolute minimum support 3, eight frequent itemsets are expected and were confirmed on the CPU reference: bread(4), milk(4), diaper(4), beer(3), bread+milk(3), bread+diaper(3), milk+diaper(3), and beer+diaper(3).'),
 P('Additional CPU tests use 100 randomized small datasets checked against an independent subset-enumeration oracle. The CUDA host-simulator code path was tested separately, but a simulator is not proof of actual NVIDIA kernel execution. When performing lab benchmarks, run the strict itemset/support verifier before timing, and retain its terminal log.'),
 S('2.1 Experiment design'),
 P('Each of the ten input files is evaluated at four decreasing support thresholds. The thresholds are intentionally dataset-specific to obtain feasible yet informative pattern counts. The program records execution time including input parsing and mining, FP-tree metrics, patterns, maximum pattern length, process peak resident memory, and CUDA buffer high-water when available. Higher pattern counts generally increase work, but memory and runtime also depend on transaction length, unique items, tree compression and number of conditional-tree builds.'),
 S('2.2 Avoiding misleading comparisons'),
 P('CPU and CUDA times are compared only when measured on the same machine in the same script run; the plotted ratio is CPU median time / CUDA median time. A value greater than 1 means CUDA was faster, a value less than 1 means CPU was faster. GPU device buffer high-water excludes driver and context allocations and is NOT directly equivalent to peak GPU process memory. The program timer starts before input parsing and ends after mining; it excludes Python startup and bulk CSV pattern writing (--no-output), but includes CUDA launch/transfer overhead as measured inside the C++ binary.'),
 P('<b>Dataset selection:</b> UCI Iris, UCI Wine, UCI Wisconsin Diagnostic Breast Cancer, UCI Handwritten Digits (all discretized as documented), plus synthetic Grocery, Bundles, Clickstream, Sensors, Long-tail, and Retail 50k. The six synthetic files are for stress testing and are not described as UCI benchmarks.'),
 PageBreak()]

short=['Iris','Wine','WDBC','Digits','Grocery','Bundles','Clickstream','Sensors','Long-tail','Retail 50k']
dataset_table=[['# / dataset','Origin','Rows','Support sweep','Patterns at lowest']]
for i,ds in enumerate(CATALOG):
    rs=by_file[ds['filename']]
    dataset_table.append([f'{i+1}. {short[i]}','UCI-derived' if i<4 else 'Synthetic',rs[0]['transactions'],
                          f'{rs[0]["min_support"]} to {rs[-1]["min_support"]}',f'{int(rs[-1]["frequent_itemsets"]):,}'])
story += [H('3. The ten datasets and experimental settings'),
 table(dataset_table,[113,89,63,114,124]),
 Spacer(1,.12*inch),
 P('Every file above is part of the project ZIP. Data origins, transformations and reproducibility notes are provided in data/TEN_DATASET_SOURCES.md. The support settings are fixed in scripts/benchmark_sweep.py so repeating the experiment does not require editing the report.'),
 P('<b>What is shown next:</b> each dataset receives a separate page with CPU runtime and CUDA runtime (when measured), number of frequent itemsets, memory, and speedup against minimum support; a table provides the exact measurements behind every plotted point.'),
 PageBreak()]

for i,ds in enumerate(CATALOG):
    rs=by_file[ds['filename']]
    story.append(H(f'{4+i}. Dataset analysis: {ds["label"]}'))
    img=Image(str(FIGURES[ds['filename']]),width=490,height=490*6.5/9.0)
    story.append(img)
    header=['Min sup','Itemsets','CPU ms','CUDA ms','CPU MiB','Speedup']
    d=[header]
    for r in rs:
        d.append([r['min_support'],f'{int(r["frequent_itemsets"]):,}',f'{float(r["cpu_median_ms"]):.3f}',
                  f'{float(r["cuda_median_ms"]):.3f}' if all_have_gpu else 'Pending',
                  f'{float(r["cpu_peak_ram_mib"]):.2f}',
                  f'{float(r["speedup_cpu_over_cuda"]):.2f}x' if all_have_gpu else 'Pending'])
    story.append(table(d,[79,84,81,81,87,91]))
    lo=rs[-1];hi=rs[0];ratio=int(lo['frequent_itemsets'])/max(1,int(hi['frequent_itemsets']))
    commentary=f'Lowering support from {hi["min_support"]} to {lo["min_support"]} changed the frequent itemset count from {int(hi["frequent_itemsets"]):,} to {int(lo["frequent_itemsets"]):,}. At the lowest support, the CPU median runtime was {float(lo["cpu_median_ms"]):.2f} ms and peak host RSS was {float(lo["cpu_peak_ram_mib"]):.2f} MiB.'
    if all_have_gpu:
        sp=float(lo['speedup_cpu_over_cuda'])
        commentary+=f' CUDA median time was {float(lo["cuda_median_ms"]):.2f} ms, corresponding to {sp:.2f}x CPU/CUDA speedup (GPU {"faster" if sp>1 else "slower" if sp<1 else "equal"}).'
    else:
        commentary+=' NVIDIA CUDA benchmarks are pending; no GPU performance claim is made.'
    story.extend([Spacer(1,10),P(TXT(commentary),'SmallX'),PageBreak()])

story += [H('14. Discussion and conclusions'),
 P('On the prepared datasets, decreasing the support threshold increases or preserves the number of frequent itemsets, as required mathematically. Runtime does not have to increase monotonically because of measurement noise and differences in tree traversal, but for most datasets a lower threshold makes more itemsets and potentially more conditional trees. The sensitivity to support is why a single support setting cannot characterize FP-Growth performance.'),
 P('This implementation retains FP-tree construction, recursive traversal and conditional-base management on the CPU, so GPU acceleration is confined to repeated weighted histograms. It differs materially from a GPU-resident FP-Growth design. Many small CUDA launches can dominate end-to-end execution even if histogram operations parallelize well. Conclusions about GPU speedup should be based on the 40 actual same-machine measurements, not performance figures from another implementation or GPU.'),
 S('Limitations and future work'),
 P('The prepared ten-dataset bundle comprises four UCI-derived files and six synthetic workloads. A requirement for ten original UCI datasets would need six replacements before submission. The chosen thresholds are not uniform and were selected for feasible runtime, so cross-dataset timing should be interpreted within each dataset rather than as a controlled comparison of difficulty. Small datasets have timing noise. A fully GPU-resident implementation or conditional-pattern-base batching could change the performance balance, but neither is claimed as implemented.'),
 S('References'),
 P('Han, J., Pei, J. and Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372','SmallX'),
 P('UCI Machine Learning Repository. https://archive.ics.uci.edu/','SmallX'),
 P('NVIDIA CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/','SmallX'),
 PageBreak(),H('15. Lab commands and submission checklist'),
 Preformatted('''make
make cuda
python3 scripts/verify_cuda.py --random-cases 20
python3 scripts/benchmark_sweep.py --with-cuda --repeats 3
python3 scripts/generate_sweep_report.py''',ss['CodeX2']),
 P('Install PDF dependencies through Ubuntu system packages if they are not available: sudo apt install python3-reportlab python3-matplotlib. Do not use sudo pip, especially on a shared lab workstation. Alternatively create a Python virtual environment if permitted.'),
 P('Save the strict comparison output as a log, verify that the report explicitly names the NVIDIA GPU and prints CPU/GPU speedup for all forty settings, inspect all ten charts, and only then print the PDF and email your own source code, datasets and report in the ZIP. Ensure the assignment due date is confirmed with the instructor.'),
 P('<b>Source note:</b> This report is generated from results/support_sweep_results.csv created by this project. The comparison project supplied by a classmate was used to identify useful experimental presentation practices, not as a source for benchmark numbers or as code to copy.','SmallX')]
OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=letter,leftMargin=.72*inch,rightMargin=.72*inch,
                     topMargin=.72*inch,bottomMargin=.67*inch,title='CUDA FP-Growth - Ten Dataset Support Sweep',author='Student project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(f'Created {OUT} ({len(rows)} dataset/threshold measurements)')
print('Report CUDA results:', 'Measured on '+str(META_DEVICE) if all_have_gpu else 'PENDING - requires lab benchmark_sweep.py --with-cuda')
