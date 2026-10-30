#!/usr/bin/env python3
"""Generate the UPDATED multi-threshold PDF report (10 datasets x 4 supports).

Uses measured results/support_sweep_results.csv, not numbers from a reference project.
"""
import runpy
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'scripts' / 'generate_sweep_report.py'), run_name='__main__')
src = ROOT / 'report' / 'FP_Growth_Expanded_10_Dataset_Report.pdf'
dst = ROOT / 'report' / 'FP_Growth_CUDA_10_Dataset_Report.pdf'
shutil.copy2(src,dst)
print('Updated default report path:',dst)
