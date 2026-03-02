from __future__ import annotations
import numpy as np
import pandas as pd

def recommend_budget_from_roi(channels, base_spend, roi_curve_fn, total_budget, bounds, step=0.02):
    ch = channels[:]
    mins = np.array([bounds[c][0] for c in ch])
    maxs = np.array([bounds[c][1] for c in ch])
    steps = int(round(1/step))
    best = None

    for i in range(int(mins[0]*steps), int(maxs[0]*steps)+1):
        for j in range(int(mins[1]*steps), int(maxs[1]*steps)+1):
            f0=i/steps; f1=j/steps
            rem=1.0-f0-f1
            if rem<=0: continue
            rest=np.array([base_spend.get(c,1.0) for c in ch[2:]], dtype=float)
            w = rest/rest.sum() if rest.sum()>0 else np.ones_like(rest)/len(rest)
            fracs=np.concatenate([[f0,f1], rem*w])
            if np.any(fracs < mins-1e-9) or np.any(fracs > maxs+1e-9):
                continue
            alloc={c: fracs[k]*total_budget for k,c in enumerate(ch)}
            val=sum(roi_curve_fn(c, alloc[c]) for c in ch)
            if (best is None) or (val>best['value']):
                best={'alloc':alloc,'value':float(val)}

    rows=[]
    for c in ch:
        rows.append({
            'channel': c,
            'current_spend': float(base_spend.get(c,0.0)),
            'recommended_spend': float(best['alloc'][c]),
            'delta': float(best['alloc'][c]-base_spend.get(c,0.0)),
        })
    out=pd.DataFrame(rows).sort_values('recommended_spend', ascending=False).reset_index(drop=True)
    out.attrs['objective_value']=best['value']
    return out
