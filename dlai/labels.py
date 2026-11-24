"""Gene-tree window labels and the transition anchor (no data fit of the transition).

Labels (window_labels): T1 and T3 windows are Background; T2 windows are split into INT and ILS. The INT volume is the
T2 excess in bp (bp of T2 windows minus bp of T3 windows, supported bp only); T2 windows are ranked by rank(f_d) +
rank(donor-private site count) (ties by f_d, then at random with a fixed seed) and the top-ranked ones are INT until
their summed bp is closest to the volume (at least one window when the volume is below the first window); INT windows
with fewer than DP_MIN donor-private sites become ILS (the top window stays INT). Four-state labels: T3 windows (part of
Background in the three-state labels) are ILS_SD.

Anchor (anchor_theta, anchor_q4): dwell times = mean bp length of the labelled runs (supported bp; map-effective bp with
a genetic map), exit splits = observed run-to-run transitions. Three-state anchor (used by the diagnostics): the ILS
dwell and exits are tied to the T3 runs and the entries into ILS read from the T3 runs per bp of source. Four-state
generator (the decode): Background, INT, ILS_RD, ILS_SD; ILS_RD and ILS_SD leave like the T3 runs, Background enters
ILS_RD and ILS_SD at the observed T3-entry rate, INT exits to T3 are split evenly between ILS_RD and ILS_SD."""
import numpy as np
import pandas as pd

from .io import unsupported_overlap

DP_MIN = 2.0
OFFDIAG = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]   # theta order: O->I, O->L, I->O, I->L, L->O, L->I


def topo_stats(gt):
    """window counts: pi_int = max(0, (T2 - T3) / resolved windows), prop_ils = T3 / resolved windows."""
    t = gt.topology.values
    n2, n3 = int((t == "T2").sum()), int((t == "T3").sum())
    n = int(np.isin(t, ["T1", "T2", "T3"]).sum())
    return dict(pi_int=max(0.0, (n2 - n3) / max(n, 1)), prop_ils=n3 / max(n, 1))


def alphabet_codes(alph):
    """alphabet code of the ABBA group, BABA, AABA, ABAA (for f_d per window)."""
    default = dict(ABBA=0, AABA=1, ABAA=2, BABA=3)
    ci = {}
    for name in ("ABBA", "AABA", "ABAA", "BABA"):
        found = None
        for i, lab in enumerate(alph.labels):
            if (name == "ABBA" and "ABBA" in str(lab).split("+")) or lab == name:
                found = i; break
        ci[name] = found if found is not None else default[name]
    return ci


def rank_windows(fd, cdp, span, target):
    """The label ranking: windows ordered by rank(f_d) + rank(donor-private count), ties by f_d then at random (seed
    0), best first; k = the number of top windows whose summed bp is closest to target. -> (order, k, cumulative bp)."""
    fdv = np.nan_to_num(fd, nan=-np.inf)
    rk = lambda v: pd.Series(v).rank(method="average").values
    o = np.lexsort((np.random.default_rng(0).random(len(fdv)), fdv, rk(fdv) + rk(cdp)))[::-1]
    cum = np.cumsum(span[o])
    k = int(np.searchsorted(cum, target, side="left"))
    if k < len(o) and abs(cum[k] - target) < abs((cum[k - 1] if k else 0.0) - target):
        k += 1
    return o, k, cum


def window_labels(pos, ref_codes, gt, ci, dp_site, unsup, log=print):
    """-> dict(gt = window table (+ bp_acc with unsupported spans), wstate (per window: -1 none, 0 Bg, 1 INT, 2 ILS),
    w4 (per window, T3 -> 3), ws / ws4 (per site), widx (site -> window or -1), window_bp (table to write or None))."""
    gt = gt.reset_index(drop=True)
    starts, ends, topo = gt.start_pos.values, gt.end_pos.values, gt.topology.values
    span = (ends - starts + 1).astype(float)
    window_bp = None
    if unsup is not None:
        raw = span
        span = np.maximum(span - unsupported_overlap(starts, ends, unsup), 0.0)
        gt = gt.assign(bp_acc=span)
        window_bp = gt[["start_pos", "end_pos", "topology", "bp_acc"]]
        log(f"[unsupported] {(raw - span).sum():,.0f} of {raw.sum():,.0f} gene-tree window bp are unsupported and not "
            f"counted; {int((raw >= 1e6).sum())} window(s) span >= 1 Mb raw")
    widx = np.searchsorted(starts, pos, side="right") - 1
    ok = (widx >= 0) & (widx < len(gt))
    inside = ok & (pos <= ends[np.clip(widx, 0, len(gt) - 1)])
    widx = np.where(inside, widx, -1)
    m = widx >= 0
    c = {}
    for k in ("ABBA", "BABA", "AABA", "ABAA"):
        c[k] = np.zeros(len(gt))
        np.add.at(c[k], widx[m], (ref_codes[m] == ci[k]))
    den = c["ABBA"] + c["ABAA"] + c["AABA"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fd = np.where(den > 0, (c["ABBA"] - c["BABA"]) / den, np.nan)
    cdp = np.zeros(len(gt))
    md = m & np.asarray(dp_site, bool)
    np.add.at(cdp, widx[md], 1.0)
    valid = np.isin(topo, ["T1", "T2", "T3"]); is_t2 = topo == "T2"
    bp_valid, bp_t2, bp_t3 = span[valid].sum(), span[is_t2].sum(), span[topo == "T3"].sum()
    target = max(bp_t2 - bp_t3, 0.0)
    pi_i, pi_l = max(1e-3, target / max(bp_valid, 1)), max(1e-3, bp_t3 / max(bp_valid, 1))
    wstate = np.full(len(gt), -1); wstate[np.isin(topo, ["T1", "T3"])] = 0
    t2i = np.flatnonzero(is_t2)
    o, k, cum = rank_windows(fd[t2i], cdp[t2i], span[t2i], target)
    if k == 0 and len(o):
        k = 1; pi_i = max(1e-3, cum[0] / max(bp_valid, 1))
        log(f"[labels] 0 INT windows at target {target:,.0f} bp -> top T2 window ({cum[0]:,.0f} bp)")
    wstate[t2i] = 2; wstate[t2i[o[:k]]] = 1
    log(f"[labels] T2 {bp_t2:,.0f} bp, T3 {bp_t3:,.0f} bp of {bp_valid:,.0f} -> INT target {target:,.0f} bp: INT = {k} of "
        f"{len(t2i)} T2 windows ({cum[k - 1] if k else 0:,.0f} bp); pi_I={pi_i:.4f} pi_L={pi_l:.4f}")
    d = (wstate == 1) & (cdp < DP_MIN); n1 = int((wstate == 1).sum())
    if n1 and int(d.sum()) == n1:
        d[t2i[o[0]]] = False
    wstate[d] = 2
    log(f"[labels] {int(d.sum())} of {n1} INT windows ({span[d].sum():,.0f} of {cum[k - 1] if k else 0:,.0f} bp) have "
        f"< {DP_MIN:g} donor-private sites -> ILS")
    w4 = wstate.copy(); w4[topo == "T3"] = 3
    ws = np.full(len(pos), -1); ws[m] = wstate[widx[m]]
    ws4 = np.full(len(pos), -1); ws4[m] = w4[widx[m]]
    return dict(gt=gt, wstate=wstate, w4=w4, ws=ws, ws4=ws4, widx=widx, window_bp=window_bp)


def _runs_bp(lab, start, end, val):
    out, i, n = [], 0, len(lab)
    while i < n:
        if lab[i] == val:
            j = i
            while j + 1 < n and lab[j + 1] == val:
                j += 1
            out.append(float(end[j]) - float(start[i]) + 1.0); i = j + 1
        else:
            i += 1
    return np.asarray(out, float)


def _runs_acc(lab, acc, val):
    out, i, n = [], 0, len(lab)
    while i < n:
        if lab[i] == val:
            j = i
            while j + 1 < n and lab[j + 1] == val:
                j += 1
            out.append(float(np.sum(acc[i:j + 1]))); i = j + 1
        else:
            i += 1
    return np.asarray(out, float)


def _labelled_windows(gt, lab_all, gmap):
    """the labelled windows (supported bp > 0), their labels, and a run-length function in (effective) bp"""
    if "bp_acc" in gt.columns:
        keep = (lab_all >= 0) & (gt.bp_acc.values > 0); acc = gt.bp_acc.values[keep]
    else:
        keep = lab_all >= 0; acc = None
    lab, st, en = lab_all[keep], gt.start_pos.values[keep], gt.end_pos.values[keep]
    if gmap is not None:
        raw = (en - st + 1).astype(float)
        st_e = gmap.eff(st); en_e = gmap.eff(en + 1) - 1.0
        if acc is not None:
            acc = acc * np.maximum(en_e - st_e + 1.0, 1.0) / np.maximum(raw, 1.0)
        st, en = st_e, en_e
    if acc is None:
        runs = lambda lb, z: _runs_bp(lb, st, en, z)
        wspan = float(np.mean(en - st + 1)) if len(st) else 1e4
    else:
        runs = lambda lb, z: _runs_acc(lb, acc, z)
        wspan = float(np.mean(acc)) if len(acc) else 1e4
    return keep, lab, runs, wspan


def anchor_theta(gt, wstate, gmap=None, log=print):
    """Three-state anchor (Other, INT, ILS) -> 6 log-rates (OFFDIAG order)."""
    keep, lab, runs, wspan = _labelled_windows(gt, wstate, gmap)
    dw = [runs(lab, z) for z in (0, 1, 2)]
    dwell = np.array([d.mean() if len(d) else wspan for d in dw])
    log(f"[anchor] dwell bp: Other {dwell[0]:,.0f} ({len(dw[0])} runs)  INT {dwell[1]:,.0f} ({len(dw[1])})  "
        f"ILS {dwell[2]:,.0f} ({len(dw[2])})")
    cnt = np.zeros((3, 3))
    for a, b in zip(lab[:-1], lab[1:]):
        if a != b:
            cnt[a, b] += 1
    Q = np.zeros((3, 3))
    for i in range(3):
        s = cnt[i].sum() - cnt[i, i]
        for j in range(3):
            if j != i:
                Q[i, j] = max((cnt[i, j] / s) if s > 0 else 0.5, 1e-6) / dwell[i]
        Q[i, i] = -Q[i].sum()
    # ILS dwell and exits from the T3 runs
    tp = gt.topology.values[keep]
    lab4 = np.where((lab == 0) & (tp == "T3"), 3, lab)
    d3 = runs(lab4, 3); d3m = d3.mean() if len(d3) else dwell[2]
    ex = np.zeros(3)
    for x, y in zip(lab4[:-1], lab4[1:]):
        if x == 3 and y != 3:
            ex[y] += 1
    tot = 1.0 / d3m; spl = ex / ex.sum() if ex.sum() else np.array([0.5, 0.25, 0.25])
    r = dwell[2] / d3m
    Q[:, 2] *= r; Q[2, :] = 0.0
    Q[2, 0] = tot * max(spl[0] + spl[2], 1e-6); Q[2, 1] = tot * max(spl[1], 1e-6)
    for i in range(3):
        Q[i, i] = 0.0; Q[i, i] = -Q[i].sum()
    # entries into ILS from the T3 runs, per bp of source; INT dwell kept
    ent = np.zeros(3)
    for x, y in zip(lab4[:-1], lab4[1:]):
        if y == 3 and x != 3:
            ent[x] += 1
    bp4 = np.array([runs(lab4, z).sum() for z in range(4)])
    Q[0, 2] = max((ent[0] + ent[2]) / max(bp4[0] + bp4[2], 1.0), 1e-6 / dwell[0])
    Q[1, 2] = min(max(ent[1] / max(bp4[1], 1.0), 1e-6 / dwell[1]), (1.0 - 1e-6) / dwell[1])
    Q[1, 0] = 1.0 / dwell[1] - Q[1, 2]
    for i in range(3):
        Q[i, i] = 0.0; Q[i, i] = -Q[i].sum()
    log(f"[anchor] ILS tied to T3 runs: T3 dwell {d3m:,.0f} bp, exit split T1/INT/ILS {spl.round(3).tolist()}, entries "
        f"T1/INT/ILS {ent.astype(int).tolist()}")
    return np.array([np.log(Q[i, j]) for (i, j) in OFFDIAG], float)


def anchor_q4(gt, w4, gmap=None, log=print):
    """Four-state generator (Background, INT, ILS_RD, ILS_SD)."""
    keep, lab, runs, wspan = _labelled_windows(gt, w4, gmap)
    rl = [runs(lab, z) for z in range(4)]
    dwell = np.array([r.mean() if len(r) else wspan for r in rl]); bp = np.array([r.sum() for r in rl])
    cnt = np.zeros((4, 4))
    for a, b in zip(lab[:-1], lab[1:]):
        if a != b:
            cnt[a, b] += 1
    Q = np.zeros((4, 4))
    ex = cnt[3, [0, 1, 2]]; spl = ex / ex.sum() if ex.sum() else np.array([0.5, 0.25, 0.25])
    dS = dwell[3] if len(rl[3]) else dwell[2]
    Q[3, 0], Q[3, 1], Q[3, 2] = (max(spl[k], 1e-6) / dS for k in range(3))
    Q[2, 0], Q[2, 1], Q[2, 3] = Q[3, 0], Q[3, 1], Q[3, 2]
    s0 = cnt[0].sum()
    Q[0, 1] = max((cnt[0, 1] / s0) if s0 > 0 else 0.5, 1e-6) / dwell[0]
    Q[0, 3] = Q[0, 2] = max(cnt[0, 3] / max(bp[0], 1.0), 1e-6 / dwell[0])
    dI = dwell[1]; e0, e3 = cnt[1, 0], cnt[1, 3]
    f3 = min(max(e3 / (e0 + e3) if e0 + e3 > 0 else 0.5, 1e-6), 1.0 - 1e-6)
    Q[1, 0] = (1.0 - f3) / dI; Q[1, 3] = Q[1, 2] = 0.5 * f3 / dI
    for i in range(4):
        Q[i, i] = 0.0; Q[i, i] = -Q[i].sum()
    nm = ["Bg", "INT", "RDT", "SDT"]
    log(f"[anchor4] label runs Bg/INT/RDT/SDT {[len(r) for r in rl]}, dwell bp " + "/".join(f"{d:,.0f}" for d in dwell)
        + f"; T3 exit split Bg/INT/RDT {spl.round(3).tolist()}; INT exits to T1/T3 {int(e0)}/{int(e3)}")
    for i in range(4):
        log(f"[anchor4]   {nm[i]:<4} -> " + "  ".join(f"{nm[j]} {Q[i, j]:.3e}" for j in range(4) if j != i)
            + f"   (dwell {-1.0 / Q[i, i]:,.0f} bp)")
    return Q
