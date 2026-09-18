#!/usr/bin/env python3
"""Run one predeclared algebra-level feasibility study and write complete results."""
from __future__ import annotations
import argparse
import importlib
import platform
import resource
import subprocess
import time
import numpy as np
import scipy
from common import write_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction',required=True,choices=['commutator','gram','spectral'])
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--smoke',action='store_true',help='development only; not a research result')
    args=parser.parse_args()
    start=time.perf_counter()
    rows,metadata=importlib.import_module(args.direction).run(args.seed,args.smoke)
    metadata.update({'direction':args.direction,'seed':args.seed,'smoke':args.smoke,
                     'elapsed_s':time.perf_counter()-start,'python':platform.python_version(),
                     'numpy':np.__version__,'scipy':scipy.__version__,
                     'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    write_results(args.output_dir,rows,metadata)
    print(f'Completed {args.direction}: {len(rows)} rows, {metadata["elapsed_s"]:.3f} s',flush=True)

if __name__=='__main__':
    main()
