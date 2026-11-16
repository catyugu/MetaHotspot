#!/usr/bin/env python3
"""Run paired extractions sequentially, then validate and summarize."""
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sequence=[('fresh',20260805),('shift_cache',20260805),('port_cache',20260805),
          ('shift_cache',20260806),('fresh',20260806),('port_cache',20260806),
          ('port_cache',20260807),('fresh',20260807),('shift_cache',20260807)]
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    env[key]='1'
output=HERE/'results'
for method,seed in sequence:
    target=output/f'{method}_{seed}.json'
    if target.exists():
        print('Resume: keeping',target.name,flush=True)
        continue
    with (HERE/f'{method}_{seed}.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'benchmark.py'),'--method',method,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed',target.name,flush=True)
subprocess.run([sys.executable,str(HERE/'evaluate.py')],env=env,check=True)
subprocess.run([sys.executable,str(HERE/'summarize.py')],env=env,check=True)

for script in sorted((HERE/'figures').glob('make_fig_*.py')):
    subprocess.run([sys.executable,str(script)],env=env,check=True)
