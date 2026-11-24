"""Per-site design-effect evidence weights, built from the first (unweighted, no map) decode; truth is never read.

w_i = 1 / (1 + sum_{0 < |j - i| <= K} rho(d_ij))
  runs   maximal stretches of consecutive sites called INT (P(INT) >= 0.5) or not INT
  r_i    log-LR (INT vs Other, three-state emission) minus the mean log-LR of its run
  rho(d) correlation of r_i, r_j over same-run pairs at distance d (lags 1..K sites), in NBIN log-spaced distance bins
         (+ one bin from 0 to the 0.1% quantile of site gaps); bins with >= MIN_PAIRS pairs only (a sparse bin takes the
         value of the nearest estimable shorter bin); made non-increasing by weighted isotonic regression (weights =
         pairs); 0 beyond the last estimable bin with rho >= 2 SE (SE = 1/sqrt(pairs))
  d      genetic distance in cM with a genetic map (linear interpolation between map points), else bp"""
import os

import numpy as np
import pandas as pd

K = 200
NBIN = 40
MIN_PAIRS = 1000


def pava_dec(y, w):
    """weighted isotonic regression, non-increasing"""
    y = list(-np.asarray(y, float)); w = list(np.asarray(w, float)); blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            a, b = blocks.pop(), blocks.pop()
            ww = a[1] + b[1]; blocks.append([(a[0] * a[1] + b[0] * b[1]) / ww, ww, a[2] + b[2]])
    out = []
    for v, _, n in blocks:
        out += [-v] * n
    return np.array(out)


def rho_curve(x, r, run, edges):
    v = r.var(); S = np.zeros(len(edges) - 1); N = np.zeros(len(edges) - 1)
    for k in range(1, K + 1):
        same = run[k:] == run[:-k]
        if not same.any():
            break
        d = (x[k:] - x[:-k])[same]; p = (r[k:] * r[:-k])[same]
        b = np.searchsorted(edges, d, side="right") - 1; ok = (b >= 0) & (b < len(S))
        S += np.bincount(b[ok], weights=p[ok], minlength=len(S)); N += np.bincount(b[ok], minlength=len(S))
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = S / (N * v)
    ok = N >= MIN_PAIRS
    sm = np.zeros(len(S))
    if not ok.any():
        return raw, sm, N
    sm[ok] = pava_dec(raw[ok], N[ok])
    se = 1 / np.sqrt(np.maximum(N, 1))
    iok = np.flatnonzero(ok)
    strong = iok[sm[iok] >= 2 * se[iok]]
    end = strong[-1] + 1 if len(strong) else 0
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(S)), -1)); idx[idx < 0] = iok[0]
    sm = sm[idx]
    sm[end:] = 0.0
    return raw, np.clip(sm, 0, None), N


def deff_sites(x, edges, rho):
    n = len(x); tot = np.ones(n)
    for k in range(1, K + 1):
        if k >= n:
            break
        d = x[k:] - x[:-k]
        b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, None)
        val = np.where(b < len(rho), rho[np.minimum(b, len(rho) - 1)], 0.0)
        tot[:-k] += val; tot[k:] += val
    return tot


def recipient_weights(ell, p_int, x):
    lab = (p_int >= 0.5).astype(int)
    run = np.r_[0, np.cumsum(np.diff(lab) != 0)]
    r = ell - pd.Series(ell).groupby(run).transform("mean").values
    dpos = np.diff(x); dpos = dpos[dpos > 0]
    lo = max(np.quantile(dpos, 0.001), 1e-12) if len(dpos) else 1e-12
    hi = np.max(x[K:] - x[:-K]) if len(x) > K else np.max(x) - np.min(x)
    edges = np.r_[0.0, np.geomspace(lo, hi * 1.0001, NBIN)]
    raw, rho, N = rho_curve(x, r, run, edges)
    dsite = deff_sites(x, edges, rho)
    w = 1.0 / dsite
    rows = [dict(lo=edges[i], hi=edges[i + 1], n=N[i], raw=raw[i], rho=rho[i]) for i in range(len(rho))]
    summ = dict(n_sites=len(x), int_share=float(lab.mean()), w_mean=float(w.mean()), w_median=float(np.median(w)),
                w_lt25=float((w < 0.25).mean()), rho0=float(rho.max()), nbins_pos=int((rho > 0).sum()),
                mean_deff=float(dsite.mean()))
    return w, rows, summ


def build(per_recipient, out, genetic_map=None, log=print):
    """per_recipient: {recipient: (pos, log-LR, P(INT))} in recipient order. Writes <out>/<recipient>.tsv (pos, w at 6
    significant digits), summary.tsv, rho.tsv, rule.txt. -> {recipient: w as written}."""
    unit = "cM" if genetic_map else "bp"
    log(f"[weights] design-effect weights, distance in {unit}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rule.txt"), "w") as fh:
        fh.write(f"distance={unit}\ncut=last\nmap={os.path.abspath(genetic_map) if genetic_map else ''}\n")
    if genetic_map:
        M = pd.read_csv(genetic_map, sep=r"\s+", header=None, usecols=[2, 3], names=["cm", "bp"]).sort_values("bp")
    rows, rhos, out_w = [], [], {}
    for rc in sorted(per_recipient):
        pos, ell, p_int = per_recipient[rc]
        if not np.isfinite(ell).all():
            raise SystemExit(f"{rc}: non-finite log-LR")
        x = np.interp(pos.astype(float), M.bp.values, M.cm.values) if genetic_map else pos.astype(float)
        w, rr, summ = recipient_weights(ell, p_int, x)
        f = os.path.join(out, f"{rc}.tsv")
        pd.DataFrame({"pos": pos, "w": w}).to_csv(f, sep="\t", index=False, float_format="%.6g")
        out_w[rc] = pd.read_csv(f, sep="\t")["w"].values.astype(float)
        rows.append(dict(recip=rc, distance=unit, **summ)); rhos += [dict(recip=rc, **q) for q in rr]
        log(f"  {rc}: mean w {summ['w_mean']:.3f} (median {summ['w_median']:.3f}, {summ['w_lt25']:.0%} < .25, "
            f"{summ['nbins_pos']} distance bins with rho > 0)")
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(rhos).to_csv(os.path.join(out, "rho.tsv"), sep="\t", index=False, float_format="%.5g")
    return out_w
