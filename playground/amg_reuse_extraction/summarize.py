#!/usr/bin/env python3
"""Collate measured runs; never fill absent results with estimates."""
import json
from pathlib import Path
import statistics
import hashlib
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
METHODS=['fresh','port_cache','shift_cache']
SEEDS=[20260805,20260806,20260807]


def summarize(output=HERE/'results'):
    raw=[json.loads((output/f'{m}_{seed}.json').read_text()) for m in METHODS for seed in SEEDS]
    validation=json.loads((output/'validation.json').read_text())
    fields=['max_cached_hierarchies','total_s','setup_count','response_count','cg_iterations','max_linear_residual','peak_rss_kib']
    def stats(values):
        return dict(mean=statistics.mean(values),std=statistics.stdev(values),n_seeds=len(values),values=values)
    summary={}
    for method in METHODS:
        rows=[row for row in raw if row['method']==method]
        summary[method]={key:stats([row[key] for row in rows]) for key in fields}
        summary[method]['phase_s']={key:stats([row['times'][key] for row in rows]) for key in rows[0]['times']}
        summary[method]['basis_orders']=[row['summary']['basis_order'] for row in rows]
        vals=[row for row in validation['runs'] if row['method']==method]
        summary[method]['accuracy']={key:stats([row['worst'][key] for row in vals]) for key in vals[0]['worst']}
    paired={}
    for method in METHODS[1:]:
        values=[]
        for seed in SEEDS:
            a=next(r for r in raw if r['method']=='fresh' and r['seed']==seed)
            b=next(r for r in raw if r['method']==method and r['seed']==seed)
            values.append(a['total_s']/b['total_s'])
        paired[method]=stats(values)
    basis_identity={}
    for seed in SEEDS:
        digests=[]
        for method in ('port_cache','shift_cache'):
            with (output/f'{method}_{seed}_basis.npy').open('rb') as handle:
                digests.append(hashlib.file_digest(handle,'sha256').hexdigest())
        basis_identity[str(seed)]={'port_cache_sha256':digests[0],'shift_cache_sha256':digests[1],'identical':digests[0]==digests[1]}
    sources={str(path.relative_to(HERE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob('*.py'))}
    result=dict(status='completed',task='End-to-end parametric ROM extraction with same-frequency AMG reuse',
                seeds=SEEDS,methods=METHODS,provenance=dict(mode='measured',simulated=False,
                source='results/{method}_{seed}.json and results/validation.json, actual CPU numerical experiment',
                timestamp=datetime.now(timezone.utc).isoformat(),git_sha=raw[0]['environment']['git_sha'],
                ci_artifact_id=14121435121,source_hashes=sources),summary=summary,paired_speedup=paired,
                ablation=dict(port_cache='Reuse across HTC within each port; no cross-port reuse'),
                environment=raw[0]['environment'],validation=validation,basis_identity=basis_identity)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    for method in METHODS:
        s=summary[method]
        print(method,'seconds',s['total_s'],'setups',s['setup_count']['values'],'responses',s['response_count']['values'],'orders',s['basis_orders'])
    print('paired speedups',paired)
    print('reference max true residual',validation['reference_max_true_residual'])
    return result

if __name__=='__main__': summarize()
