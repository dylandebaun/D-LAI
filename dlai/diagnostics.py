"""Diagnostics (reported from the first, unweighted decode; never used to change calls).

diagnostics.tsv        per recipient: topology fractions and T2 excess (windows and bp), T2/T3 linkage ratio, gene-tree
                       distance ratios, anchor pi / dwell, het_share, excess JS INT-ILS / INT-Other, % called INT
gate.tsv               R = mean JS_INT_ILS_excess / mean JS_INT_Other_excess (the structure ratio; reported only)
diag/                  model refits on the decode's sites (B0 / B / Tc / To and a dwell x entry profile; refit_models)
diagnostics_trial.tsv  trial statistics (trial_diagnostics): structure R / R6 with block-jackknife SE, rate ratio,
                       divergence ratios in INT vs ILS tracts, D / f_hom / f_d, data summary"""
import os
import shutil

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .hmm import q_from_theta, stationary
from .labels import anchor_theta

DWELL_MULTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ENTRY_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
R6_CODES = np.array([l in "BH" and p == "0" for l in "ABH" for p in "0123" for _ in "012"])


def _js(p, q):
    m = (p + q) / 2
    kl = lambda x, y: float((x * np.log2(x / y)).sum())
    return (kl(p, m) + kl(q, m)) / 2


# ---------------------------------------------------------------------------------------------------------- per run
def window_table_bp(g):
    """supported bp of each window (bp_acc when the run has unsupported spans), else the raw span."""
    raw = (g.end_pos - g.start_pos + 1).values.astype(float)
    if "bp_acc" not in g.columns:
        return raw
    v = g.bp_acc.values.astype(float)
    return np.where(np.isfinite(v), v, raw)


def recipient_table(D, recs, theta, tables, counts):
    """diagnostics.tsv. theta: {recipient: three-state anchor log-rates}; tables: {recipient: window table};
    counts: {recipient: count DataFrame (code-indexed, the recipient's OWN counts)}."""
    rng = np.random.default_rng(1); rows = []
    for r in recs:
        f = dict(recip=r)
        g = tables[r]
        g = g[g.topology.isin(["T1", "T2", "T3"])].reset_index(drop=True); tp = g.topology.values
        for k in ("T1", "T2", "T3"):
            f[f"frac_{k}"] = (tp == k).mean()
        f["T2_excess"] = f["frac_T2"] - f["frac_T3"]
        span = window_table_bp(g)
        for k in ("T1", "T2", "T3"):
            f[f"frac_bp_{k}"] = span[tp == k].sum() / span.sum()
        f["T2_excess_bp"] = f["frac_bp_T2"] - f["frac_bp_T3"]
        b = np.flatnonzero(np.r_[True, tp[1:] != tp[:-1], True]); L = np.diff(b); lab = tp[b[:-1]]
        lk = {k: (L[lab == k].mean() * (1 - (tp == k).mean())) if (lab == k).any() else np.nan for k in ("T2", "T3")}
        f["linkage_ratio_T2_T3"] = lk["T2"] / lk["T3"]
        t1, t2, t3 = g.med_d12.values[tp == "T1"], g.med_d23.values[tp == "T2"], g.med_d13.values[tp == "T3"]
        f["bl_ratio_T2d23_T1d12"] = np.median(t2) / np.median(t1) if len(t1) and len(t2) and np.median(t1) > 0 else np.nan
        f["bl_short_excess"] = ((t2 < np.quantile(t3, 0.10)).sum() - 0.10 * len(t3)) / len(tp) if len(t3) > 5 else np.nan
        if r in theta:
            Q = q_from_theta(theta[r])
            w, v = np.linalg.eig(Q.T); pi = np.real(v[:, np.argmin(np.abs(w))]); pi /= pi.sum(); dw = -1 / np.diag(Q)
            f.update(anchor_pi_INT=pi[1], anchor_pi_ILS=pi[2], dwell_INT_bp=dw[1], dwell_ILS_bp=dw[2],
                     dwell_ratio_INT_ILS=dw[1] / dw[2])
        q = counts[r]; mu = lambda c: (c + 0.5) / (c + 0.5).sum()
        q = q[~q.index.str.contains("M")]
        cI, cL, cO = q.n_INT.values, q.n_ILS.values, q.n_Other.values
        hz, hm = q.n_INT[["H01", "H02"]].sum(), q.n_INT[["B01", "B02"]].sum()
        f.update(n_INT_sites=cI.sum(), het_share=hz / (hz + hm) if hz + hm else np.nan)
        for a_, b_, nm in ((cI, cL, "INT_ILS"), (cI, cO, "INT_Other")):
            na, nb = int(round(a_.sum())), int(round(b_.sum())); pool = (a_ + b_) / max((a_ + b_).sum(), 1e-9)
            null = np.mean([_js(mu(rng.multinomial(na, pool)), mu(rng.multinomial(nb, pool))) for _ in range(100)]) \
                if na and nb else np.nan
            f[f"JS_{nm}_excess"] = _js(mu(a_), mu(b_)) - null
        tf = os.path.join(D, f"{r}_tracts.csv")
        if os.path.exists(tf):
            t = pd.read_csv(tf); t = t[t.state >= 0]
            ln_ = (t.end - t.start + 1).astype(float)
            f.update(pct_called=100 * ln_[t.state == 1].sum() / ln_.sum(), n_int_tracts=int((t.state == 1).sum()))
        rows.append(f)
    pd.DataFrame(rows).to_csv(os.path.join(D, "diagnostics.tsv"), sep="\t", index=False)


def gate(D, n_recip, log=print):
    """gate.tsv: the structure ratio R (reported; calls are never changed)."""
    dg = pd.read_csv(os.path.join(D, "diagnostics.tsv"), sep="\t")
    jl, jo = dg.JS_INT_ILS_excess.mean(), dg.JS_INT_Other_excess.mean()
    ratio = float(jl / jo) if np.isfinite(jl) and np.isfinite(jo) and jo > 0 else float("nan")
    pd.DataFrame([dict(JS_INT_ILS_excess_mean=jl, JS_INT_Other_excess_mean=jo, ratio=ratio, threshold=0.0, flagged=0,
                       n_recip=n_recip)]).to_csv(os.path.join(D, "gate.tsv"), sep="\t", index=False)
    log(f"[gate] R = JS_INT_ILS {jl:.4f} / JS_INT_Other {jo:.4f} = {ratio:.4f} (reported only)")


def adopt_unweighted(D, U):
    """The reported diagnostics are the unweighted decode's; the weighted decode's own table is kept as
    diagnostics_weighted.tsv, and pct_called / n_int_tracts are recomputed from the final (weighted) tracts."""
    if os.path.exists(os.path.join(D, "diagnostics.tsv")):
        os.replace(os.path.join(D, "diagnostics.tsv"), os.path.join(D, "diagnostics_weighted.tsv"))
    for f in ("diagnostics.tsv", "diagnostics_trial.tsv"):
        if os.path.exists(os.path.join(U, f)):
            shutil.copy(os.path.join(U, f), os.path.join(D, f))
    f = os.path.join(D, "diagnostics.tsv")
    if os.path.exists(f):
        dg = pd.read_csv(f, sep="\t")
        for i, r in dg.recip.items():
            tf = os.path.join(D, f"{r}_tracts.csv")
            if os.path.exists(tf):
                t = pd.read_csv(tf); t = t[t.state >= 0]; ln_ = (t.end - t.start + 1).astype(float)
                dg.loc[i, "pct_called"] = 100 * ln_[t.state == 1].sum() / ln_.sum()
                dg.loc[i, "n_int_tracts"] = int((t.state == 1).sum())
        dg["diagnostics_from"] = "unweighted decode (calls: weighted)"
        dg.to_csv(f, sep="\t", index=False)


def donor_private_per_tract(D, vg, p1, p2, p3, og, log=print):
    """Annotate <recip>_tracts.csv with n_donor_private / donor_private_per_kb: sites where the recipient carries a
    derived allele, no P1-panel member does (all called) and the P3 panel does (outgroup called hom-REF). Reported
    only; a low count marks a weakly determined call whatever its posterior."""
    G, sm, pos = vg["G"], vg["samples"], vg["pos"]
    if any(x not in sm for x in list(p1) + list(p3) + [og]):
        log("[donor-private] samples missing from the VCF read; skipped"); return
    nalt = lambda names: G[:, [sm.index(x) for x in names]].to_n_alt(fill=-1)
    n1, n3 = nalt(list(p1)), nalt(list(p3))
    p1_lacks = (n1 >= 0).all(1) & (np.where(n1 > 0, n1, 0).sum(1) == 0)
    p3_has = np.where(n3 > 0, n3, 0).sum(1) > 0
    keep = G[:, sm.index(og)].to_n_alt(fill=-1) == 0
    for r in p2:
        f = os.path.join(D, f"{r}_tracts.csv")
        if not os.path.exists(f) or r not in sm:
            continue
        g = G[:, sm.index(r)].to_n_alt(fill=-1)
        cs = np.r_[0, np.cumsum((g > 0) & p1_lacks & p3_has & keep)]
        t = pd.read_csv(f)
        a = np.searchsorted(pos, t.start.values); b = np.searchsorted(pos, t.end.values, side="right")
        t["n_donor_private"] = cs[b] - cs[a]
        t["donor_private_per_kb"] = (t.n_donor_private / ((t.end - t.start + 1) / 1e3)).round(4)
        t.to_csv(f, index=False)
    log(f"[donor-private] annotated {len(p2)} tract file(s)")


def read_vcf_genotypes(vcf, samples, contig=None):
    """Every VCF record (scikit-allel), all samples (donor-private annotation) and the run's samples in VCF order
    (trial statistics); only `contig` of a multi-contig VCF (needs its tabix index)."""
    import allel
    v = allel.read_vcf(vcf, fields=["samples", "variants/POS", "calldata/GT"], region=contig)
    sm = list(v["samples"]); gt = v["calldata/GT"]
    sub = [s for s in sm if s in set(samples)]
    return dict(pos=v["variants/POS"], samples=sm, G=allel.GenotypeArray(gt),
                sub=sub, Gs=gt[:, [sm.index(s) for s in sub]])


# ------------------------------------------------------------------------------------- in-core model refits (diag/)
def _batch_fb(log_emits, emit_idx, thetas, didx, unique_dists, post=()):
    """Scaled forward (+ backward for the models in post) of several three-state CTMC HMMs at once."""
    M, n = len(thetas), log_emits[0].shape[1]; uds = np.asarray(unique_dists, float); idx = np.asarray(emit_idx)
    Qs = [q_from_theta(np.asarray(t, float)) for t in thetas]
    try:
        T = np.stack([expm(Q[None] * uds[:, None, None]) for Q in Qs])
    except Exception:
        T = np.stack([np.stack([expm(Q * d) for d in uds]) for Q in Qs])
    np.clip(T, 0.0, None, out=T)
    lmax = np.stack([le.max(0) for le in log_emits])
    E = np.stack([np.exp(le - le.max(0)) for le in log_emits])
    pi = np.stack([stationary(Q) for Q in Qs]); pi = pi / pi.sum(1, keepdims=True)
    post = list(post); P = np.array(post, int)
    la = np.empty((len(post), 3, n)) if post else None; cs = np.empty((M, n))
    a = pi * E[idx, :, 0]; c = a.sum(1); a = a / c[:, None]; cs[:, 0] = c
    if post:
        la[:, :, 0] = a[P]
    for t in range(1, n):
        a = np.einsum("mi,mij->mj", a, T[:, didx[t]]) * E[idx, :, t]
        c = a.sum(1); a = a / c[:, None]; cs[:, t] = c
        if post:
            la[:, :, t] = a[P]
    ll = np.log(cs).sum(1) + lmax[idx].sum(1)
    gam = {}
    if post:
        b = np.ones((len(post), 3)); TP = T[P]; EP = E[idx[P]]; csP = cs[P]
        g = np.empty((len(post), 3, n)); g[:, :, n - 1] = la[:, :, n - 1]
        for t in range(n - 2, -1, -1):
            b = np.einsum("mij,mj->mi", TP[:, didx[t + 1]], EP[:, :, t + 1] * b) / csP[:, t + 1][:, None]
            g[:, :, t] = la[:, :, t] * b
        g /= g.sum(1, keepdims=True)
        gam = {m: g[k] for k, m in enumerate(post)}
    return ll, gam


def _tie_mu(cnt):
    mu = cnt / cnt.sum(1, keepdims=True)
    sh = np.array([c < 36 and (c % 12) // 3 > 0 for c in range(cnt.shape[1])])
    pool = cnt[1, sh] + cnt[2, sh]; pool = pool / pool.sum()
    for z in (1, 2):
        mu[z, sh] = mu[z, sh].sum() * pool
    return mu


def refit_models(pre, p2, ctx, theta0, labels, log_emit_A, distances, ll_A, n_blocks=20, log=print):
    """Refits on the decode's own sites (A = the decode's three-state model): B0 (no INT, ILS absorbs nothing), B (INT
    labels moved to ILS), Tc (INT and ILS emissions pooled), To (INT and Other pooled) and a profile over ILS dwell x
    entry multipliers. -> <pre>_<p2>_{diag,profile,blocks}.tsv."""
    code, ws, cnt, muA, tw = ctx["code"], ctx["ws"], ctx["cnt"], ctx["mu"], ctx["w"]
    ok = code >= 0; cd = np.where(ok, code, 0)

    def lem(mu):
        le = np.zeros((3, len(code))); le[:, ok] = np.log(mu)[:, cd[ok]]
        return le if tw is None else le * tw[None, :]
    fin = np.isfinite(log_emit_A)
    emit_chk = float(np.abs(lem(muA) - log_emit_A)[fin].max()) if fin.any() else np.nan
    uds = np.unique(distances); didx = np.searchsorted(uds, distances)
    th0 = np.asarray(theta0, float); Q0 = q_from_theta(th0)
    dI, dL = -1 / Q0[1, 1], -1 / Q0[2, 2]
    off = lambda th: np.r_[np.log(1e-12), th[1:5], np.log(1e-12)]
    mu_chk = float(np.abs(_tie_mu(cnt) - muA).max())
    muTc = muA.copy(); p = cnt[1] + cnt[2]; muTc[1] = p / p.sum(); muTc[2] = muTc[1].copy()
    muTo = muA.copy(); p = cnt[0] + cnt[1]; muTo[0] = p / p.sum(); muTo[1] = muTo[0].copy()
    cB = cnt.copy(); cB[2] = cnt[2] + (cnt[1] - 0.5); cB[1] = 0.5; muB = _tie_mu(cB)
    gt_l, wst = labels
    thB = anchor_theta(gt_l, np.where(wst == 1, 2, wst), None, log=lambda *a: None)
    grid, gth = [], []
    for dm in sorted(set(DWELL_MULTS) | {round(dI / dL, 6)}):
        for pm in ENTRY_MULTS:
            th = th0.copy(); k = dI / (dm * dL)
            for i in (0, 2, 3, 5):
                th[i] += np.log(k)
            th[0] += np.log(pm); th[5] += np.log(pm); grid.append((dm, pm)); gth.append(th)
    grid.append((0.0, 0.0)); gth.append(off(th0))
    ems = [log_emit_A, lem(muTc), lem(muTo), lem(muB)]
    names = ["A", "B0", "Tc", "To", "B"]; thetas = [th0, off(th0), off(th0), off(th0), off(thB)]; eidx = [0, 0, 1, 2, 3]
    ll, gam = _batch_fb(ems, eidx + [0] * len(gth), thetas + gth, didx, uds, post=(1, 4))
    L = dict(zip(names, ll[:5])); prof = ll[5:]
    rows = [dict(dwell_mult_ils=dm, entry_mult=pm, logL=v) for (dm, pm), v in zip(grid, prof)]
    for r, th in zip(rows, gth):
        Q = q_from_theta(th); r["int_dwell_bp"] = -1 / Q[1, 1]; r["pi_int"] = stationary(Q)[1]
    pr = pd.DataFrame(rows)[["dwell_mult_ils", "entry_mult", "int_dwell_bp", "pi_int", "logL"]]
    pr.to_csv(f"{pre}_{p2}_profile.tsv", sep="\t", index=False)
    q = pr[pr.entry_mult > 0]; best = q.logL.max()
    cost_short = best - q[q.dwell_mult_ils >= 1].logL.max(); cost_long = best - q[q.dwell_mult_ils <= 1].logL.max()
    m2 = ok & (ws == 2); ct = np.zeros(int(m2.sum()))
    for z in (0, 2):
        ct += (gam[1][z, m2] + gam[4][z, m2]) / 2 * (np.log(muA[z, cd[m2]]) - np.log(muB[z, cd[m2]])) \
            * (1.0 if tw is None else tw[m2])
    lab = np.flatnonzero(ok & (ws >= 0)); blk = np.minimum((np.arange(len(lab)) * n_blocks) // max(len(lab), 1), n_blocks - 1)
    C = np.zeros((n_blocks, 3, cnt.shape[1])); np.add.at(C, (blk, ws[lab], code[lab]), 1)
    bi, ci = np.nonzero(C.sum(1))
    pd.DataFrame(dict(block=bi, code=ci, n_Other=C[bi, 0, ci], n_INT=C[bi, 1, ci], n_ILS=C[bi, 2, ci])).to_csv(
        f"{pre}_{p2}_blocks.tsv", sep="\t", index=False)
    out = dict(recip=p2, n_sites=len(code), n_coded=int(ok.sum()),
               **{f"n_lab_{s}": int((ok & (ws == z)).sum()) for z, s in ((0, "Other"), (1, "INT"), (2, "ILS"))},
               logL_A=ll_A, logL_A_batch=L["A"], logL_B0=L["B0"], logL_B=L["B"], logL_Tc=L["Tc"], logL_To=L["To"],
               profile_best=best, cost_short=cost_short, cost_long=cost_long, ILSwin_sum=float(ct.sum()), ILSwin_n=len(ct),
               int_dwell_bp=dI, ils_dwell_bp=dL, anchor_pi_int=float(stationary(Q0)[1]), mu_refit_maxdiff=mu_chk,
               emit_A_maxdiff=emit_chk, diag_site_w_mean=(float(tw.mean()) if tw is not None else 1.0))
    pd.DataFrame([out]).to_csv(f"{pre}_{p2}_diag.tsv", sep="\t", index=False)
    log(f"[diag] {p2}: logL A {ll_A:.3f} (batch {L['A']:.3f}) B0 {L['B0']:.3f} B {L['B']:.3f} Tc {L['Tc']:.3f} "
        f"To {L['To']:.3f}; profile cost short {cost_short:.3f} long {cost_long:.3f}")


# --------------------------------------------------------------------------------------------- trial statistics
def _jsx(a, b, rng, nnull=100):
    mu = lambda c: (c + 0.5) / (c + 0.5).sum(); keep = (a + b) > 0; a, b = a[keep], b[keep]
    na, nb = int(round(a.sum())), int(round(b.sum()))
    if not na or not nb:
        return np.nan
    pool = (a + b) / (a + b).sum()
    return _js(mu(a), mu(b)) - np.mean([_js(mu(rng.multinomial(na, pool)), mu(rng.multinomial(nb, pool))) for _ in range(nnull)])


def _ratio_R(counts, mask, rng, nnull):
    il, io = [], []
    for C in counts:
        C = C[:, mask]; il.append(_jsx(C[1], C[2], rng, nnull)); io.append(_jsx(C[1], C[0], rng, nnull))
    a, b = np.nanmean(il) if il else np.nan, np.nanmean(io) if io else np.nan
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan, a, b


def _structure(D, recs, nblock=20):
    blocks = {}
    for r in recs:
        f = os.path.join(D, "diag", f"dlai_{r}_blocks.tsv")
        if os.path.exists(f):
            t = pd.read_csv(f, sep="\t"); B = np.zeros((nblock, 3, 36))
            t = t[t.code < 36]
            B[t.block.values, 0, t.code.values] = t.n_Other.values; B[t.block.values, 1, t.code.values] = t.n_INT.values
            B[t.block.values, 2, t.code.values] = t.n_ILS.values; blocks[r] = B
    out = {}
    if not blocks:
        return out
    for nm, mask in (("full", np.ones(36, bool)), ("R6", R6_CODES)):
        rng = np.random.default_rng(1)
        R, a, b = _ratio_R([B.sum(0) for B in blocks.values()], mask, rng, 100)
        loo = np.array([_ratio_R([B.sum(0) - B[k] for B in blocks.values()], mask, rng, 20)[0] for k in range(nblock)])
        se = float(np.sqrt((nblock - 1) / nblock * np.nansum((loo - np.nanmean(loo)) ** 2)))
        out.update({f"R_{nm}_blocks": R, f"SE_{nm}": se, f"JSx_INT_ILS_{nm}": a, f"JSx_INT_Other_{nm}": b})
    return out


def _rr_from_gt(g):
    t1 = g.med_d12[g.topology == "T1"].median()
    if not np.isfinite(t1) or t1 <= 0:
        return np.nan
    r = np.array([g.med_d23[g.topology == "T2"].median() / t1, g.med_d13[g.topology == "T3"].median() / t1], float)
    return float(np.nanmin(r)) if np.isfinite(r).any() else np.nan


def _dxy_cum(G, i, j):
    x, y = G[:, i, :], G[:, j, :]; ok = (x >= 0).all(1) & (y >= 0).all(1)
    d = sum((x[:, a] != y[:, b]) for a in (0, 1) for b in (0, 1)) / 4
    return np.r_[0.0, np.cumsum(np.where(ok, d, 0.0))]


def _divergence(pos, sm, G, tr, rc, p1, p3, og, fixed=None):
    """fixed: the recipient-independent cumulative dxy (P1-P3, P3-outgroup), computed once per trial"""
    ix = [sm.index(x) for x in (rc, p3, p1, og)]
    C13, C3o = fixed if fixed is not None else (_dxy_cum(G, ix[2], ix[1]), _dxy_cum(G, ix[1], ix[3]))
    C23, C12 = _dxy_cum(G, ix[0], ix[1]), _dxy_cum(G, ix[0], ix[2])
    t = tr.copy(); i0, i1 = np.searchsorted(pos, t.start.values), np.searchsorted(pos, t.end.values, side="right")
    for k, Cc in (("d23", C23), ("d13", C13), ("d12", C12), ("d3o", C3o)):
        t[k] = Cc[i1] - Cc[i0]
    t["bp"] = t.end - t.start + 1
    a = t.groupby("state")[["d23", "d13", "d12", "d3o", "bp"]].sum()
    o = dict(int_bp=float(a.bp.get(1, 0)))
    per = lambda s, k: a.loc[s, k] / a.loc[s, "bp"] if s in a.index and a.loc[s, "bp"] > 0 else np.nan
    for k in ("23", "13", "12"):
        o[f"Q{k}"] = per(1, f"d{k}") / per(2, f"d{k}") if per(2, f"d{k}") else np.nan
    o["Q23n"] = (a.loc[1, "d23"] / a.loc[1, "d3o"]) / (a.loc[2, "d23"] / a.loc[2, "d3o"]) \
        if 1 in a.index and 2 in a.index and a.loc[1, "d3o"] > 0 and a.loc[2, "d3o"] > 0 and a.loc[2, "d23"] > 0 else np.nan
    o["dxr_INT_ILS_P1P2"] = o["Q12"]
    o["dxr_INT_Other_P2P3"] = per(1, "d23") / per(0, "d23") if per(0, "d23") else np.nan
    return o


def _panel_freq(sm, G, names):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(G[:, [sm.index(x) for x in names], :] >= 0,
                                   G[:, [sm.index(x) for x in names], :], np.nan).reshape(len(G), -1), 1)


def _f_stats(sm, G, p1, rc, p3, og, fixed=None):
    """fixed: the recipient-independent allele frequencies (P1 panel, P3, outgroup), computed once per trial"""
    a1, a3, ao = fixed if fixed is not None else (_panel_freq(sm, G, p1), _panel_freq(sm, G, [p3]), _panel_freq(sm, G, [og]))
    a2 = _panel_freq(sm, G, [rc])
    ok = np.isfinite(a1 + a2 + a3 + ao); a1, a2, a3, ao = a1[ok], a2[ok], a3[ok], ao[ok]
    fl = ao > 0.5; a1, a2, a3 = np.where(fl, 1 - a1, a1), np.where(fl, 1 - a2, a2), np.where(fl, 1 - a3, a3)
    q = np.where(fl, 1 - ao, ao)
    abba = lambda x, y, z: (1 - x) * y * z * (1 - q); baba = lambda x, y, z: x * (1 - y) * z * (1 - q)
    num = (abba(a1, a2, a3) - baba(a1, a2, a3)).sum(); den = (abba(a1, a2, a3) + baba(a1, a2, a3)).sum()
    fh = (abba(a1, a3, a3) - baba(a1, a3, a3)).sum(); pdd = np.maximum(a2, a3); fd = (abba(a1, pdd, pdd) - baba(a1, pdd, pdd)).sum()
    return dict(D=num / den if den else np.nan, f_hom=num / fh if fh else np.nan, f_d=num / fd if fd else np.nan)


def trial_diagnostics(D, vg, p1, p2, p3, og, tables, log=print):
    """Adds per-recipient statistics to diagnostics.tsv and writes diagnostics_trial.tsv (one row)."""
    dg = pd.read_csv(os.path.join(D, "diagnostics.tsv"), sep="\t")
    pos, sm, G = vg["pos"], vg["sub"], vg["Gs"]
    p1v = [x for x in p1 if x in sm]
    fx_div = fx_f = None
    if p1v and p3[0] in sm and og in sm:
        i1, i3, io = sm.index(p1v[0]), sm.index(p3[0]), sm.index(og)
        fx_div = (_dxy_cum(G, i1, i3), _dxy_cum(G, i3, io))
        fx_f = (_panel_freq(sm, G, p1v), _panel_freq(sm, G, [p3[0]]), _panel_freq(sm, G, [og]))
    rows = []
    for r in p2:
        f = dict(recip=r)
        df = os.path.join(D, "diag", f"dlai_{r}_diag.tsv")
        if os.path.exists(df):
            f.update(pd.read_csv(df, sep="\t").iloc[0].drop("recip").to_dict())
        g = tables.get(r)
        f["rr_min"] = _rr_from_gt(g) if g is not None else np.nan
        tf = os.path.join(D, f"{r}_tracts.csv")
        if os.path.exists(tf) and r in sm and p1v and p3[0] in sm and og in sm:
            f.update(_divergence(pos, sm, G, pd.read_csv(tf), r, p1v[0], p3[0], og, fx_div))
            f.update(_f_stats(sm, G, p1v, r, p3[0], og, fx_f))
        sf = os.path.join(D, f"{r}_sites.csv")
        if g is not None and os.path.exists(sf):
            sp = pd.read_csv(sf, usecols=["pos"]).pos.values
            n = np.searchsorted(sp, g.end_pos.values, side="right") - np.searchsorted(sp, g.start_pos.values)
            f.update(n_sites_decoded=len(sp), n_gt_windows=len(g), med_sites_per_window=float(np.median(n)) if len(n) else np.nan)
        rows.append(f)
    X = pd.DataFrame(rows)
    keep = [c for c in dg.columns if c not in X.columns or c == "recip"]
    dg = dg[keep].merge(X, on="recip", how="left"); dg.to_csv(os.path.join(D, "diagnostics.tsv"), sep="\t", index=False)
    T = dict(n_recip=len(p2), n_vcf_sites=int(len(pos)), call_rate=float((G >= 0).all(2).mean()) if G.size else np.nan)
    m = lambda c: float(dg[c].mean()) if c in dg and dg[c].notna().any() else np.nan
    s = lambda c: float(dg[c].sum()) if c in dg and dg[c].notna().any() else np.nan
    jl, jo = m("JS_INT_ILS_excess"), m("JS_INT_Other_excess")
    T.update(R_full=jl / jo if np.isfinite(jl) and np.isfinite(jo) and jo > 0 else np.nan, JS_INT_ILS=jl, JS_INT_Other=jo)
    T.update(_structure(D, p2))
    T["R6"] = T.get("R_R6_blocks", np.nan)
    for nm, c in (("R", "R_full"), ("R6", "R6")):
        for th in (0.15, 0.17):
            T[f"would_fire_{nm}_{th:.2f}"] = int(np.isfinite(T[c]) and T[c] <= th)
    for c in ("frac_T1", "frac_T2", "frac_T3", "T2_excess", "frac_bp_T1", "frac_bp_T2", "frac_bp_T3", "T2_excess_bp",
              "anchor_pi_INT", "pct_called", "rr_min", "Q23", "Q13", "Q12", "Q23n", "dxr_INT_ILS_P1P2", "dxr_INT_Other_P2P3",
              "D", "f_hom", "f_d", "med_sites_per_window", "n_sites_decoded", "het_share", "bl_ratio_T2d23_T1d12"):
        T[c] = m(c)
    T["rr_signed"] = -T["rr_min"] if np.isfinite(T["T2_excess"]) and T["T2_excess"] < 0 else T["rr_min"]
    A, B0, B, Tc, To = (s(f"logL_{k}") for k in ("A", "B0", "B", "Tc", "To"))
    cs, cl = s("cost_short"), s("cost_long")
    T.update(logL_A=A, logL_B0=B0, logL_B=B, logL_Tc=Tc, logL_To=To, A_Tc=A - Tc, A_To=A - To, A_B0=A - B0, B0_B=B0 - B,
             r_other=(A - To) / (A - Tc) if A != Tc else np.nan, b=(B0 - B) / (A - B0) if A != B0 else np.nan,
             l=cs / (A - Tc) if A != Tc else np.nan, l_star=cs / (cs + abs(A - Tc)) if cs + abs(A - Tc) > 0 else np.nan,
             l_long=cl / (cl + abs(A - Tc)) if cl + abs(A - Tc) > 0 else np.nan, cost_short=cs, cost_long=cl,
             ILSwin=s("ILSwin_sum") / s("ILSwin_n") if s("ILSwin_n") else np.nan,
             logL_A_batch_diff=float((dg.logL_A - dg.logL_A_batch).abs().max()) if "logL_A_batch" in dg else np.nan)
    fast = np.isfinite(T["rr_min"]) and T["rr_min"] < 0.8
    T["fast_lineage"] = ("P2" if T["T2_excess"] < 0 else "P1") if fast else ""
    T["decision"] = ("zero" if (np.isfinite(T["l_star"]) and T["l_star"] > 0.13) else "warn") if fast else "report"
    f0 = np.isfinite(T["rr_min"]) and T["rr_min"] < 0.5
    hom = T["ILSwin"] < 0 and T["b"] <= -0.5 and T["l"] > 0.15
    T["decision_v0"] = ("zero" if hom else "warn") if f0 else "report"
    pd.DataFrame([T]).to_csv(os.path.join(D, "diagnostics_trial.tsv"), sep="\t", index=False, float_format="%.10g")
    log(f"[diagnostics] R {T['R_full']:.4f} (SE {T.get('SE_full', np.nan):.4f})  R6 {T['R6']:.4f}  rr_min "
        f"{T['rr_min']:.3f}  l* {T['l_star']:.3f}  decision {T['decision']}")
