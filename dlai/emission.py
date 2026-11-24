"""Per-state emission: one code per site (sites.emission_codes), a multinomial per state fitted from the window labels.

counts      training counts per state (Other / INT / ILS from the three-state labels; ILS_SD = the T3-labelled part of
            Other), weighted by the per-site evidence weight in the weighted decode
pooling     with several recipients the emission is fitted from the counts summed over every recipient (each recipient's
            own sites counted once), so every recipient gets the same cohort emission
mu          (counts + 0.5) normalised per state; INT and ILS are tied on the codes where the P1 panel carries the
            derived allele (their summed mass is shared out by the pooled INT + ILS counts)
four states Background = Other minus ILS_SD, INT, ILS_RD = ILS, ILS_SD
copy number (annotation) INT, ILS and ILS_SD are each split into one- and two-copy components (int2_split) when the
            two-copy share is at least WMIN and the state has at least NMIN training sites"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .sites import code_label

WMIN, NMIN = 0.02, 200.0       # copy-number split: minimum two-copy share and training sites
FGRID = (1.0,)                 # inbreeding of the two-copy genotype model


def training_counts(code, nc, ws, ws4, w=None):
    """-> (raw counts (3, nc) per three-state label, raw ILS_SD counts (nc,)), weighted by w if given."""
    raw = np.zeros((3, nc))
    for z in (0, 1, 2):
        m = ws == z
        raw[z] = np.bincount(code[m], weights=None if w is None else w[m], minlength=nc)
    m3 = ws4 == 3
    t3 = np.bincount(code[m3], weights=None if w is None else w[m3], minlength=nc).astype(float)
    return raw, t3


def counts_table(nc, raw, t3, mu=None):
    """DataFrame code, [mu_*], n_Other, n_INT, n_ILS, n_T3 (the mu_<recipient>.tsv layout). raw = cnt - 0.5 of fit()
    (not the bincount itself: the pooled counts are built from these values)."""
    d = {"code": [code_label(c) for c in range(nc)]}
    if mu is not None:
        d.update(mu_Other=mu[0], mu_INT=mu[1], mu_ILS=mu[2])
    d.update(n_Other=raw[0], n_INT=raw[1], n_ILS=raw[2], n_T3=t3)
    return pd.DataFrame(d)


def pool_tables(tables):
    """Sum per-recipient count tables (code-indexed, recipient order) -> pooled table (code index)."""
    acc = None
    for t in tables:
        d = t.set_index("code").filter(items=["n_Other", "n_INT", "n_ILS", "n_T3"])
        acc = d if acc is None else acc.add(d, fill_value=0)
    return acc


def _int2_geno(qd, qo, F):
    """genotype letter probabilities (A, B, H) for one introgressed copy (donor allele + background) and two copies."""
    p1 = np.stack([(1 - qd) * (1 - qo), qd * qo, qd * (1 - qo) + (1 - qd) * qo])
    p2 = np.stack([(1 - F) * (1 - qd) ** 2 + F * (1 - qd), (1 - F) * qd ** 2 + F * qd, (1 - F) * 2 * qd * (1 - qd)])
    return p1, p2


def _int2_bins(nc):
    if nc == 36:
        return np.arange(36).reshape(3, 12)
    return np.array([[l * 12 + b for b in range(12)] + [36 + l * 8 + o for o in range(8)] for l in range(3)])


def int2_split(mu, cnt, z=1):
    """Split state z's emission into one- and two-copy components by fitting the two-copy share w (and per-bin donor
    allele frequencies) to its training counts. -> dict(mu1, mu2, w, F, ll, n, w_mix)."""
    NC = mu.shape[1]; ix = _int2_bins(NC); nb = ix.shape[1]
    m = mu[:, ix]
    freq = lambda zz: (lambda pz: pz[1] + 0.5 * pz[2])(m[zz] / m[zz].sum(0, keepdims=True))
    qo = np.clip(freq(0), 1e-4, 1 - 1e-4)
    qd0 = np.clip(2.0 * freq(z) - qo, 1e-3, 1 - 1e-3)
    n = np.maximum(np.asarray(cnt, float), 0)[ix]
    lg = lambda x: np.log(x / (1 - x)); ex = lambda y: 1 / (1 + np.exp(-y))
    best = None
    for F in FGRID:
        def nll(th):
            qd, w = ex(th[:nb]), ex(th[nb])
            p1, p2 = _int2_geno(qd, qo, F)
            return -(n * np.log((1 - w) * p1 + w * p2 + 1e-300)).sum()
        r = minimize(nll, np.r_[lg(qd0), lg(0.2)], method="L-BFGS-B", bounds=[(-9, 9)] * (nb + 1))
        if best is None or r.fun < best[0]:
            best = (r.fun, F, r.x)
    _, F, th = best
    qd, w = ex(th[:nb]), float(ex(th[nb]))
    p1, p2 = _int2_geno(qd, qo, F)
    rr = np.zeros(NC); rr[ix] = (w * p2) / ((1 - w) * p1 + w * p2 + 1e-300)
    mu1 = mu[z] * (1 - rr) + 1e-9; mu2 = mu[z] * rr + 1e-9
    return dict(mu1=mu1 / mu1.sum(), mu2=mu2 / mu2.sum(), w=w, F=F, ll=-best[0], n=float(n.sum()),
                w_mix=float((mu[z] * rr).sum() / mu[z].sum()))


def fit(nc, raw, t3, pooled=None, copy_number=True, label="", log=print):
    """-> dict(cnt (3, nc incl. the 0.5 prior), mu (3, nc), mu4 (4, nc), t3, int2 (copy-number splits or None))."""
    cnt = 0.5 + raw
    labels = [code_label(c) for c in range(nc)]
    if pooled is not None:                         # cohort counts replace the recipient's own (w = 1)
        add = np.zeros((3, nc))
        for j, c in enumerate(labels):
            if c in pooled.index:
                for z, k in ((0, "n_Other"), (1, "n_INT"), (2, "n_ILS")):
                    add[z, j] = max(float(pooled.loc[c, k]), 0.0)
        own = cnt - 0.5
        cnt = 0.5 + (1.0 - 1.0) * own + 1.0 * add
        log(f"[pool] own INT {int(round(own[1].sum())):,}, cohort INT {int(round(add[1].sum())):,} -> "
            f"{int(round(cnt[1].sum() - 18)):,} INT training weight")
        addT3 = np.array([max(float(pooled.loc[c, "n_T3"]), 0.0) if c in pooled.index else 0.0 for c in labels])
        t3 = (1.0 - 1.0) * t3 + 1.0 * addT3
    mu = cnt / cnt.sum(1, keepdims=True)
    sh = np.array([c < 36 and (c % 12) // 3 > 0 for c in range(nc)])   # P1 panel carries the derived allele
    pool = cnt[1, sh] + cnt[2, sh]; pool = pool / pool.sum()
    for z in (1, 2):
        mu[z, sh] = mu[z, sh].sum() * pool
    c0 = np.maximum(cnt[0] - t3, 0.5); c3 = 0.5 + t3
    mu4 = np.vstack([c0 / c0.sum(), mu[1], mu[2], c3 / c3.sum()])
    log(f"[emission] {nc} codes; training weight Bg/INT/ILS_RD/ILS_SD = " + "/".join(f"{x:,.0f}" for x in (
        c0.sum() - 0.5 * nc, cnt[1].sum() - 0.5 * nc, cnt[2].sum() - 0.5 * nc, t3.sum())))
    int2 = None
    if copy_number:
        s = int2_split(mu, cnt[1] - 0.5); nI = float((cnt[1] - 0.5).sum())
        sL = int2_split(mu, cnt[2] - 0.5, z=2); nL = float((cnt[2] - 0.5).sum())
        log(f"[copy number] ILS two-copy share {sL['w']:.4f} from {nL:,.0f} ILS training sites"
            + (" -> no ILS split" if (sL["w"] < WMIN or nL < NMIN) else ""))
        if sL["w"] < WMIN or nL < NMIN:
            sL = None
        nS = float(np.asarray(t3, float).sum())
        sS = int2_split(np.vstack([mu[0], mu[1], mu4[3]]), t3, z=2)
        log(f"[copy number] ILS_SD two-copy share {sS['w']:.4f} from {nS:,.0f} ILS_SD training sites"
            + (" -> no ILS_SD split" if (sS["w"] < WMIN or nS < NMIN) else ""))
        if sS["w"] < WMIN or nS < NMIN:
            sS = None
        if s["w"] < WMIN or nI < NMIN:
            log(f"[copy number] INT two-copy share {s['w']:.3f} from {nI:,.0f} INT training sites -> below "
                f"{WMIN:g} / {NMIN:g}: no copy-number annotation for {label}")
        else:
            int2 = dict(s, ils=sL, sdt=sS)
            l2 = np.log(s["mu2"] / mu[2])
            log(f"[copy number] INT two-copy share {s['w']:.4f} from {nI:,.0f} INT training sites; log mu_INT2/mu_ILS at "
                + ", ".join(f"{code_label(c)}:{l2[c]:+.2f}" for c in (13, 14, 25, 26, 1, 2)))
    return dict(cnt=cnt, mu=mu, mu4=mu4, t3=t3, int2=int2, labels=labels)
