#!/usr/bin/env python3
"""Build the printable 10-dataset FP-Growth report from actual saved benchmark data.
If an actual-GPU benchmark CSV exists after lab verification, the report fills
in the GPU column automatically. Never creates invented NVIDIA benchmarks.
"""
import csv
import json
import math
from pathlib import Path
from datetime import date
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,
    Image,PageBreak,Preformatted,HRFlowable,KeepTogether)

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
OUT=ROOT/'report'/'FP_Growth_CUDA_10_Dataset_Report.pdf'
CPU=json.loads((RESULTS/'benchmark_results.json').read_text())
CATALOG=json.loads((ROOT/'scripts'/'benchmark_catalog.json').read_text())
CPUROWS=CPU['experiments']
assert len(CATALOG)==len(CPUROWS)==10
for c,r in zip(CATALOG,CPUROWS):
    assert r['file']==c['filename'], (r['file'],c['filename'])

GPUFILE=RESULTS/'cuda_benchmark_results.csv'
GPUROWS=None
if GPUFILE.exists():
    with GPUFILE.open(newline='') as f: raw=list(csv.DictReader(f))
    if len(raw)==10 and all(a['filename']==b['filename'] for a,b in zip(raw,CATALOG)) and all('SIMULATOR' not in a['gpu_device'].upper() for a in raw):
        GPUROWS=raw
    else: print('Ignoring incomplete or simulated GPU benchmark file; real GPU measurements pending.')

NAVY=colors.HexColor('#193551'); TEAL=colors.HexColor('#087C86'); BLUE=colors.HexColor('#4266A1')
SLATE=colors.HexColor('#465666'); GREY=colors.HexColor('#F1F4F7'); LINE=colors.HexColor('#D9E1E8')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverT',fontName='Helvetica-Bold',fontSize=24,leading=29,textColor=NAVY,spaceAfter=18))
styles.add(ParagraphStyle(name='CoverS',fontName='Helvetica',fontSize=12,leading=18,textColor=SLATE,spaceAfter=12))
styles.add(ParagraphStyle(name='Hdr',fontName='Helvetica-Bold',fontSize=14.4,leading=18,textColor=NAVY,spaceBefore=6,spaceAfter=10,keepWithNext=True))
styles.add(ParagraphStyle(name='Sub',fontName='Helvetica-Bold',fontSize=10.6,leading=14,textColor=TEAL,spaceBefore=9,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='Txt',fontName='Helvetica',fontSize=9.3,leading=14.4,textColor=NAVY,spaceAfter=9))
styles.add(ParagraphStyle(name='Tiny',fontName='Helvetica',fontSize=7.7,leading=11,textColor=SLATE,spaceAfter=7))
styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=7.3,leading=10,textColor=NAVY))
styles.add(ParagraphStyle(name='CellH',fontName='Helvetica-Bold',fontSize=7.2,leading=10,textColor=colors.white))
styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=7.8,leading=11.6,backColor=GREY,leftIndent=7,spaceAfter=9,spaceBefore=6))

def P(s,sty='Txt'):return Paragraph(s,styles[sty])
def H(s):return P(s,'Hdr')
def S(s):return P(s,'Sub')
def T(rows,widths):
    data=[[P(escape(str(c)),'CellH' if i==0 else 'Cell') for c in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,GREY]),
        ('LINEBELOW',(0,-1),(-1,-1),.5,LINE)
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState(); w,h=letter
    canvas.setStrokeColor(LINE)
    canvas.line(.67*inch,h-.52*inch,w-.67*inch,h-.52*inch)
    canvas.line(.67*inch,.52*inch,w-.67*inch,.52*inch)
    canvas.setFont('Helvetica',8);canvas.setFillColor(SLATE)
    canvas.drawString(.69*inch,h-.41*inch,'FP-GROWTH / CUDA HYBRID / TEN-DATASET EXPERIMENTS')
    canvas.drawString(.69*inch,.37*inch,'C++17 + NVIDIA CUDA  |  29 October 2026')
    canvas.drawRightString(w-.69*inch,.37*inch,f'{doc.page}')
    canvas.restoreState()

# Report charts. These are CPU measurements from an actually run binary.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
labels=['Iris','Wine','WDBC','Digits','Grocery','Bundles','Clickstream','Sensors','Long-tail','Retail 50k']
fig,ax=plt.subplots(figsize=(9.3,3.35),dpi=160)
vals=[r['patterns'] for r in CPUROWS]
bars=ax.bar(range(10), vals,color= ['#087C86']*4+['#4266A1']*6,width=.68)
ax.set_xticks(range(10),labels,rotation=25,ha='right');ax.set_ylabel('Frequent itemsets')
ax.grid(axis='y',alpha=.16)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+10,str(v),fontsize=7,ha='center')
ax.set_ylim(0,max(vals)*1.20)
fig.tight_layout();fig.savefig(RESULTS/'figure_10_dataset_patterns.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9.3,3.3),dpi=160)
x=[r['runtime_median_s']*1000 for r in CPUROWS]
y=[r['peak_rss_median_mib'] for r in CPUROWS]
ax.scatter(x[:4],y[:4],s=85,color='#087C86',label='UCI-derived (4)')
ax.scatter(x[4:],y[4:],s=85,color='#4266A1',label='Synthetic (6)')
for lab,xx,yy in zip(labels,x,y): ax.annotate(lab,(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.set_xlim(left=-4,right=max(x)*1.24);ax.set_ylim(bottom=0,top=max(y)*1.16)
ax.set_xlabel('CPU median execution time (milliseconds)');ax.set_ylabel('Peak process RAM (MiB)')
ax.grid(alpha=.18);ax.legend(loc='upper left')
fig.tight_layout();fig.savefig(RESULTS/'figure_10_dataset_runtime_memory.png',bbox_inches='tight');plt.close(fig)

img=lambda name,width=495:Image(str(RESULTS/(name+'.png')),width=width,height=width*(3.35/9.3))

story=[]
# Page 1
story += [Spacer(1,.42*inch),P('CUDA-Accelerated FP-Growth<br/>Ten-Dataset Benchmark Project','CoverT'),
 P('A reproducible C++17 / CUDA hybrid for complete frequent-itemset mining','CoverS'),
 HRFlowable(width='100%',thickness=2,color=TEAL,spaceBefore=9,spaceAfter=20),
 P('<b>Course:</b> ESE Computer Engineering - Frequent Itemset Mining Project<br/><b>Student:</b> Mashrafi Rodan<br/><b>Institution:</b> Stony Brook University<br/><b>Version:</b> 10 distinct offline datasets; prepared October 29, 2026'),
 Spacer(1,.14*inch),H('Abstract'),
 P('An FP-Growth miner compresses transaction records into a frequent-pattern tree (FP-tree), then recursively mines conditional pattern bases without candidate-set enumeration. The C++ implementation outputs all nonempty frequent itemsets and their exact transaction supports. A hybrid NVIDIA CUDA implementation accelerates weighted singleton support histograms at the root and for every constructed conditional FP-tree; FP-tree construction, recursion, and output remain on the CPU.'),
 P('This version includes <b>exactly ten separate offline benchmark datasets</b>: four UCI-origin copies obtained from scikit-learn with documented transformation, and six independently generated synthetic transaction datasets. All ten were benchmarked on a CPU in this preparation environment. The CPU implementation passed seven automated test groups, and all ten datasets matched a host-simulated CUDA calling path. Because no NVIDIA GPU or nvcc is available in this environment, actual CUDA compilation and GPU benchmarks must be completed in the lab.'),
 T([['Benchmark coverage','Validation status','GPU status'],['10 distinct data files\n4 UCI + 6 synthetic','7 CPU test groups passed\n10+toy+20 simulated-path checks','CUDA kernel source included\nActual NVIDIA testing pending']],[167,169,169]),
 Spacer(1,.15*inch),P('The project does not assert that the hybrid GPU version is faster. Device transfer and many short conditional-tree kernels may increase runtime. CPU/GPU speedup may be less than 1.','Tiny'),PageBreak()]

# Page 2
story += [H('1. Frequent itemset mining: definition'),
 P('Each transaction T is a set of items; multiple occurrences of the same token in one input line are deduplicated. For N transactions, support(X) counts how many transactions contain every item in X. X is frequent when support(X) is at least the absolute minimum count. Percentage thresholds become ceil(N * percentage / 100); an absolute integer or a fraction such as 0.25 is also accepted.'),
 H('2. Exact FP-Growth implementation'),
 S('2.1 Database scan and compact ordered FP-tree'),
 P('Parse all transactions, assign a numeric ID to each distinct token, and compute initial singleton frequencies. Remove items whose supports are below minimum. Sort the remaining items inside each transaction by descending global support (with item ID as a deterministic tie-breaker), then insert them into a prefix tree. Shared prefixes become shared nodes with summed counts. Header lists point to all nodes carrying a given item.'),
 S('2.2 Conditional pattern bases and recursive mining'),
 P('Process FP-tree header items from lowest support to highest. For each suffix item i, output the suffix extended by i at its exact support, walk node-to-root ancestors for every i node, and create a weighted pattern base. Each ancestor path has weight equal to the node count where i was found. Recompute frequent items using the weighted rows, construct a conditional FP-tree, and recursively enumerate extensions until the tree is empty.'),
 Preformatted('''mine(tree, suffix):
    for item in tree.items_in_ascending_support:
        pattern = suffix + item
        emit(pattern, support_in_current_tree[item])
        base = weighted_ancestor_paths(tree, item)
        if base is not empty:
            hist = support_count(base)       # CUDA in hybrid build
            next_tree = build_fp_tree(base, hist, minsup)
            if next_tree has frequent items: mine(next_tree, pattern)''',styles['CodeX']),
 S('2.3 Correctness and complexity considerations'),
 P('The weighted conditional support of an item j equals the sum of path weights over conditional rows containing j. This is the number of original transactions containing the current suffix and j. Each extension is explored only under its corresponding suffix order, so the algorithm enumerates each frequent itemset once. The output can be exponential in a dense database at low thresholds; the program supports --no-output to avoid writing gigantic CSV files.'),
 PageBreak()]

# Page 3
story += [H('3. Worked validation example'),
 P('The supplied five-basket toy dataset uses bread, milk, beer, diapers, and eggs. With minimum support 3, the program emits eight itemsets:'),
 T([['Itemset','Support'],['bread','4'],['milk','4'],['diaper','4'],['beer','3'],['bread;milk','3'],['bread;diaper','3'],['milk;diaper','3'],['beer;diaper','3']],[298,207]),
 Spacer(1,9),P('For example, milk appears in four out of five baskets. The pair bread;milk occurs in exactly three, which meets the minimum support. The independent Python oracle enumerates all subsets of each randomized small transaction and counts them directly; those results are compared against the FP-tree output.'),
 H('4. Validation already completed'),
 T([['Check','Result'],['CPU reference unit test groups','7 groups passed'],['Small random exhaustive CPU oracle','100 datasets passed'],['Ten benchmark datasets, CPU vs host-simulated CUDA path','10 / 10 complete itemset-support maps matched'],['Toy test and 20 additional random comparisons','21 / 21 exact comparisons matched'],['CUDA device-kernel compilation / real GPU equality','Not tested in preparation environment']],[368,137]),
 P('The host simulator exercises CUDA interface calls but runs counting on the CPU. Its passing results are not evidence that the actual .cu kernel compiled or executed. A separate GPU lab verifier compares the complete emitted CSV itemset/support dictionaries across both real CPU and CUDA binaries.'),
 PageBreak()]

# Page 4
story += [H('5. Ten benchmark datasets and data provenance'),
 P('Exactly ten input files are listed in scripts/benchmark_catalog.json. Each input line is one transaction. There are four genuinely UCI-derived files, provided from local scikit-learn data bundles, and six independently synthesized datasets. No classification target labels are included.'),
 T([['#','Data / source','Rows','Min. support'],*[ [str(i),d['label'],str(d['transactions']),d['support']] for i,d in enumerate(CATALOG,1)]],[29,322,80,74]),
 Spacer(1,10),S('5.1 UCI-derived datasets: preprocessing'),
 P('<b>Iris:</b> each of four continuous features is converted to one of three value bins using 1/3 and 2/3 feature quantiles. <b>Wine:</b> each of 13 numerical attributes becomes high or low relative to that feature\'s median. <b>Wisconsin Diagnostic Breast Cancer:</b> each of 30 numerical attributes is median-binarized. <b>Handwritten Digits:</b> each positive pixel intensity becomes a distinct pixel-present token. For every file, the class label is excluded.'),
 S('5.2 Synthetic datasets: intentionally different regimes'),
 P('<b>Grocery:</b> common staples; <b>Bundles:</b> planted co-occurrence groups; <b>Clickstream:</b> sparse, high-cardinality page tokens; <b>Sensors:</b> relatively dense correlated binary states; <b>Long-tail:</b> Zipf-like catalog popularity; <b>Retail 50k:</b> larger skewed transaction baskets. Seeds, data descriptions, and file SHA-256 hashes are recorded in the catalog to make the exact data reproducible.'),
 P('The other supplied toy and repeated-digits files are additional testing/scaling material and are explicitly excluded from the ten default benchmarks. Optional Mushroom/Chess/Connect-4 UCI download utilities do not count as completed offline experiments.','Tiny'),PageBreak()]

# Page 5
story += [H('6. Measured CPU results on all ten datasets'),
 P('Results below were obtained by executing the C++17 CPU reference binary in separate processes, using the median of '+str(CPU['environment'].get('timing','three trials')).split('median of ')[-1]+'. Times measure total program runtime as recorded by the executable; resident-memory peaks reflect the complete process, not just FP-tree storage. Support fractions are intentionally different across datasets, so the rows compare behavior under chosen settings, not identical difficulty.'),
 T([['Data','Support','Patterns','CPU time (ms)','Peak RAM (MiB)'], *[
     [labels[i],r['support'],f"{r['patterns']:,}",f"{r['runtime_median_s']*1000:.2f}",f"{r['peak_rss_median_mib']:.2f}"]
     for i,r in enumerate(CPUROWS)]],[151,76,84,100,94]),
 Spacer(1,14),img('figure_10_dataset_patterns',490),
 P('The number of emitted patterns varies strongly with the minimum support and feature distribution. More transactions alone do not imply more frequent itemsets.','Tiny'),PageBreak()]

# Page 6
story += [H('7. Runtime and memory interpretation'),
 img('figure_10_dataset_runtime_memory',492),
 Spacer(1,14),
 S('7.1 What the experiment indicates'),
 P('The synthetic 50,000-row retail dataset is larger and requires substantially more process memory than the smaller UCI-derived inputs. The dense-sensor dataset demonstrates that denser transactions can consume noticeable resources even with fewer transactions. Pattern count is not a sufficient predictor of elapsed time: tokenization, tree-node allocation, pointer traversal, and conditional-path construction also contribute.'),
 S('7.2 Measurement limits'),
 P('These are Linux preparation-machine CPU results, not lab GPU results. Very short runs (for instance Iris) are sensitive to clock resolution and process overhead; report their values with appropriate caution. Memory is peak resident set size reported by the running process. The overall high-water RSS can exceed the sum of actively retained FP-tree nodes because of allocator and library overhead. The CUDA buffer high-water statistic, when measured, excludes GPU driver context memory.'),
 P('<b>Test environment:</b> '+escape(CPU['environment'].get('cpu_model','unknown'))+'; '+escape(CPU['environment'].get('platform','unknown'))+'; '+escape(CPU['environment'].get('compiler','unknown'))+'.','Tiny'),
 PageBreak()]

# Page 7
story += [H('8. CUDA implementation and dataflow'),
 P('The CUDA executable is not a full GPU-resident FP-tree. Instead, it accelerates the parallel support-counting stage at EVERY FP-tree level. Host C++ flattens weighted conditional transactions into aligned item-ID and path-weight arrays. CUDA copies the arrays to device memory, clears a 64-bit histogram, launches a weighted histogram kernel with one thread per item occurrence, then copies the result back to the CPU for pruning and tree construction.'),
 Preformatted('''CPU: conditional paths -> item_ids[], path_weights[]
GPU: memcpy(H2D); zero histogram; launch kernel:
    k = blockIdx.x * blockDim.x + threadIdx.x
    if k < n: atomicAdd(counts[item_ids[k]], weights[k])
CPU: memcpy(D2H); filter, construct conditional FP-tree
CPU: recurse, producing more weighted pattern bases''',styles['CodeX']),
 T([['GPU responsibilities','CPU responsibilities'],['Root singleton histogram','Parse/normalize and deduplicate transactions'],['Weighted histograms for every conditional FP-tree','Build node/header FP-tree structure'],['64-bit atomic support aggregation','Build weighted conditional pattern bases'],['Count kernel launches and device-buffer high water','Control recursion; emit itemset/support CSV/JSON']],[245,260]),
 Spacer(1,11),S('8.1 Why CUDA might or might not speed this workload up'),
 P('Counting many independent item-weight pairs is data parallel, but each recursive conditional tree often contains few pairs. Multiple small kernel launches, memory transfers, device histogram initialization, and synchronization can outweigh the GPU arithmetic advantage. Some datasets may therefore be faster on the CPU. A fair comparison uses both binaries on the SAME lab machine, three runs each, alternating execution order, and exactly matching pattern outputs.'),
 S('8.2 Hardware validation requirement'),
 P('No CUDA GPU or nvcc is available in the report-preparation environment. Device compilation, per-dataset correctness, actual CUDA kernel call counts, device model, and CPU/GPU timing differences will be measured in the lab. Do not interpret host-simulator success as GPU validation.'),
 PageBreak()]

# Page 8
story += [H('9. Lab CUDA test and complete ten-dataset comparison'),
 Preformatted('''nvcc --version
make && make cuda
python3 scripts/verify_cuda.py --random-cases 20
python3 scripts/benchmark_cuda.py --repeats 3
python3 scripts/generate_report.py  # fills GPU results if valid''',styles['CodeX']),
 P('The benchmark driver executes all TEN datasets from benchmark_catalog.json. It refuses to silently omit a missing file. For each dataset it checks CPU/GPU pattern totals and maximum lengths, times both binaries, and records peak host RAM, device buffer high-water bytes, kernel calls, GPU device identity, and CPU-time/CUDA-time ratio. The verifier performs the stricter complete itemset/support comparison for all ten before benchmarking.'),
 T([['Data','CPU ms','CUDA ms','CPU/CUDA speedup'],*[
    [labels[i],(f"{float(GPUROWS[i]['cpu_median_seconds'])*1000:.2f}" if GPUROWS else f"{r['runtime_median_s']*1000:.2f}"),
     (f"{float(GPUROWS[i]['cuda_median_seconds'])*1000:.2f}" if GPUROWS else 'Pending lab'),
     (f"{float(GPUROWS[i]['speedup_cpu_over_gpu']):.3f}x" if GPUROWS else 'Pending lab')]
    for i,r in enumerate(CPUROWS)]],[167,105,111,122]),
 Spacer(1,10),P('<b>Note:</b> When GPU measurements are pending, CPU values come from the preparation machine. After real lab benchmarking, this table switches to the fresh CPU and GPU measurements on the SAME lab machine, making the CPU/CUDA ratio comparable.','Tiny'),
 PageBreak()]

# Page 9
story += [H('10. Discussion, limitations and conclusion'),
 P('The complete FP-Growth algorithm is implemented as actual recursive FP-tree mining. All 10 prepared datasets produce nonzero frequent itemset outputs and support data. The diverse workloads test small numeric/categorical data, moderately dense binary features, sparse large vocabularies, strongly correlated baskets, and a larger 50,000-record benchmark. The report provides exact dataset sizes and measured CPU resource usage; a separate lab step is required to establish real GPU execution and performance.'),
 P('The main limitation of this hybrid design is that FP-tree operations are irregular and remain host-side. CUDA is used for repeated integer histogram reductions, but kernel launches and PCIe transfers can dominate lightweight workloads. Future extensions could batch conditional bases from several tree branches or use a compressed GPU-friendly transaction representation; these have not been implemented.'),
 H('References and attribution'),
 P('1. Han, J., Pei, J., and Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372','Tiny'),
 P('2. UCI Machine Learning Repository: Iris (#53). https://archive.ics.uci.edu/dataset/53/iris','Tiny'),
 P('3. UCI Machine Learning Repository: Wine (#109). https://archive.ics.uci.edu/dataset/109/wine','Tiny'),
 P('4. UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) (#17). https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic','Tiny'),
 P('5. UCI Machine Learning Repository: Optical Recognition of Handwritten Digits (#80). https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','Tiny'),
 P('6. scikit-learn documentation: bundled datasets (load_iris, load_wine, load_breast_cancer, load_digits). https://scikit-learn.org/stable/datasets/toy_dataset.html','Tiny'),
 P('7. NVIDIA CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/','Tiny'),
 Spacer(1,.1*inch),H('Submission checklist'),
 P('Review course requirements; run nvcc compile and actual GPU verification in the lab; run all 10 CUDA benchmarks; regenerate the PDF and inspect CPU/GPU output; ensure the printed report includes the measured GPU results; email the final source plus PDF as a ZIP by the confirmed deadline. Do not report the host simulator as NVIDIA execution.'),
]

OUT.parent.mkdir(exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=letter,leftMargin=.72*inch,rightMargin=.72*inch,
 topMargin=.8*inch,bottomMargin=.75*inch,title='Ten Dataset CUDA-Accelerated FP-Growth',author='Mashrafi Rodan')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created '+str(OUT))
print('GPU benchmarks:', 'included from lab CSV' if GPUROWS else 'pending actual NVIDIA test')
