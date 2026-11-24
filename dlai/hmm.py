"""Continuous-time HMM along the sites: generators, forward-backward, copy-number generator, tracts.

T(d) = expm(Q d) over the (effective) bp gap d between consecutive decoded sites; pi = stationary(Q) enters at the
first site. Posteriors are the forward-backward marginals. Tract boundaries are placed inside the gap between two sites
with different calls, at the base where the posterior of the left call drops below that of the right one."""
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp

from .labels import OFFDIAG


def q_from_theta(theta):
    """3-state generator from 6 log-rates (OFFDIAG order)."""
    Q = np.zeros((3, 3))
    for r, (i, j) in zip(np.exp(np.asarray(theta, float)), OFFDIAG):
        Q[i, j] = r
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def stationary(Q):
    """normalised left null vector of Q."""
    vals, vecs = np.linalg.eig(Q.T)
    pi = np.abs(np.real(vecs[:, int(np.argmin(np.abs(vals)))]))
    s = pi.sum()
    return (pi / s) if s > 0 else np.ones(Q.shape[0]) / Q.shape[0]


def stationary_clipped(Q):
    pi = np.clip(stationary(Q), 0, None)
    return pi / pi.sum()


def log_transitions(Q, distances):
    """-> (log T stacked over the unique gaps, index of each site's gap in that stack)."""
    ud, inv = np.unique(np.asarray(distances, np.int64), return_inverse=True)
    logT = np.stack([np.log(expm(Q * int(d)) + 1e-300) for d in ud])
    return logT, inv


def lse(a, axis):
    """log-sum-exp along axis with scipy.special.logsumexp's arithmetic (maxima taken out, log1p of the rest), without
    its per-call overhead (the tract-boundary bisection calls it on 4 x 4 arrays tens of thousands of times)."""
    mx = a.max(axis=axis, keepdims=True)
    mask = a == mx
    m = mask.sum(axis=axis, keepdims=True).astype(float)
    with np.errstate(invalid="ignore"):
        s = np.where(mask, 0.0, np.exp(a - mx)).sum(axis=axis, keepdims=True)
    s = np.where(s == 0, s, s / m)
    return np.squeeze(np.log1p(s) + np.log(m) + mx, axis=axis)


try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _lse1(x):
        """scipy logsumexp arithmetic for a short vector (sequential sum, as numpy for < 8 elements)."""
        mx = -np.inf
        for v in x:
            if v > mx:
                mx = v
        if mx == -np.inf:
            return -np.inf
        m = 0.0; s = 0.0
        for v in x:
            if v == mx:
                m += 1.0
            else:
                s += np.exp(v - mx)
        if s != 0.0:
            s = s / m
        return np.log1p(s) + np.log(m) + mx

    @_nb.njit(cache=True)
    def _fb_numba(le, logT, gi):
        S, n = le.shape
        la = np.empty((S, n)); lb = np.empty((S, n)); tmp = np.empty(S)
        for j in range(S):
            la[j, 0] = le[j, 0]
        for t in range(1, n):
            T = logT[gi[t]]
            for j in range(S):
                for i in range(S):
                    tmp[i] = la[i, t - 1] + T[i, j]
                la[j, t] = _lse1(tmp) + le[j, t]
        for i in range(S):
            lb[i, n - 1] = 0.0
        for t in range(n - 2, -1, -1):
            T = logT[gi[t + 1]]
            for i in range(S):
                for j in range(S):
                    tmp[j] = T[i, j] + le[j, t + 1] + lb[j, t + 1]
                lb[i, t] = _lse1(tmp)
        return la, lb
    HAVE_NUMBA = True
except ImportError:                         # pragma: no cover
    HAVE_NUMBA = False

FAST = HAVE_NUMBA                           # compiled recursion (set False for the scipy reference loops)


def forward_backward(log_emit, distances, Q, pi):
    """log-space forward-backward. log_emit (states x sites) without pi. -> (gamma, log-likelihood, log_alpha,
    log_beta, log_emit with log(pi) added at the first site)."""
    le = log_emit.copy()
    le[:, 0] += np.log(pi + 1e-300)
    logT, gi = log_transitions(Q, distances)
    if FAST:
        la, lb = _fb_numba(np.ascontiguousarray(le), np.ascontiguousarray(logT), gi.astype(np.int64))
        lg = la + lb
        lg -= logsumexp(lg, axis=0, keepdims=True)
        return np.exp(lg), logsumexp(la[:, -1]), la, lb, le
    n_st, n = le.shape
    la = np.full((n_st, n), -np.inf); la[:, 0] = le[:, 0]
    for t in range(1, n):
        la[:, t] = logsumexp(la[:, t - 1:t] + logT[gi[t]], axis=0) + le[:, t]
    lb = np.full((n_st, n), -np.inf); lb[:, -1] = 0.0
    for t in range(n - 2, -1, -1):
        lb[:, t] = logsumexp(logT[gi[t + 1]] + le[:, t + 1] + lb[:, t + 1], axis=1)
    lg = la + lb
    lg -= logsumexp(lg, axis=0, keepdims=True)
    return np.exp(lg), logsumexp(la[:, -1]), la, lb, le


def cn_generator(Qb, ws, k=1.0):
    """Copy-number generator: each block b in ws (block -> two-copy share) gets a two-copy twin; the base rates are
    shared out by the target's copy share, the switch rate within a block is k x the block's exit rate. Lumps exactly
    to Qb. -> (Q, {block: index of its two-copy state})."""
    n0 = Qb.shape[0]
    blocks = sorted(ws)
    two = {b: n0 + i for i, b in enumerate(blocks)}
    n = n0 + len(blocks)
    members = {b: [b] + ([two[b]] if b in two else []) for b in range(n0)}
    share = np.ones(n)
    for b in blocks:
        share[b] = 1.0 - ws[b]; share[two[b]] = ws[b]
    Q = np.zeros((n, n))
    for a in range(n0):
        for b in range(n0):
            if a != b:
                for i in members[a]:
                    for j in members[b]:
                        Q[i, j] = Qb[a, b] * share[j]
    for b in blocks:
        s = k * sum(Qb[b, c] for c in range(n0) if c != b)
        Q[b, two[b]] = s * ws[b]; Q[two[b], b] = s * (1.0 - ws[b])
    for i in range(n):
        Q[i, i] = 0.0; Q[i, i] = -Q[i].sum()
    return Q, two


def collapse4(gamma4):
    """(Background, INT, ILS_RD, ILS_SD) -> (Other, INT, ILS)."""
    return np.vstack([gamma4[0], gamma4[1], gamma4[2] + gamma4[3]])


def _crossover_base(i, pos, la, lb, le, Q):
    """Base inside the gap (pos[i], pos[i+1]) where the left call's posterior falls below the right call's (bisection
    on the exact posterior at a base, no emission there)."""
    gap = int(pos[i + 1] - pos[i])
    la_i = la[:, i]
    rhs = le[:, i + 1] + lb[:, i + 1]

    def post_at(off):
        fwd = lse(la_i[:, None] + np.log(expm(Q * off) + 1e-300), 0)
        bwd = lse(np.log(expm(Q * (gap - off)) + 1e-300) + rhs[None, :], 1)
        lp = fwd + bwd
        return np.array([lp[0], lp[1], np.logaddexp(lp[2], lp[3])]) if len(lp) == 4 else lp
    a = int(post_at(0).argmax())
    b = int(post_at(gap).argmax())
    if a == b or gap <= 1:
        return int(pos[i])
    g = lambda off: (lambda p: p[a] - p[b])(post_at(off))
    if g(0) <= 0:
        return int(pos[i])
    if g(gap) >= 0:
        return int(pos[i + 1] - 1)
    lo, hi = 0, gap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return int(pos[i] + lo)


def _bp_weighted_posterior(start, end, i0, i1, pos, p):
    """mean posterior over [start, end], each site weighted by the bp of its Voronoi cell inside the tract."""
    if i0 < 0 or i1 < i0:
        return float("nan")
    if i1 == i0:
        return float(p[i0])
    lo_, hi_ = float(start), float(end) + 1.0
    num = wsum = 0.0
    for k in range(i0, i1 + 1):
        le_ = lo_ if k == i0 else 0.5 * (pos[k - 1] + pos[k])
        re_ = hi_ if k == i1 else 0.5 * (pos[k] + pos[k + 1])
        w = max(0.0, min(re_, hi_) - max(le_, lo_))
        num += w * float(p[k]); wsum += w
    if wsum <= 0.0:
        return float(np.mean([p[k] for k in range(i0, i1 + 1)]))
    return num / wsum


def tracts(pos, y_hat, p_int, la, lb, le, Q):
    """Runs of the hard call -> DataFrame(start, end, state, n_sites, mean_p_int, site_i0, site_i1, mean_p_int_bp)."""
    rows = []
    cur, cs, ps, n, i0 = int(y_hat[0]), int(pos[0]), float(p_int[0]), 1, 0
    for i in range(1, len(pos)):
        if int(y_hat[i]) == cur:
            ps += float(p_int[i]); n += 1
            continue
        bnd = _crossover_base(i - 1, pos, la, lb, le, Q)
        bnd = max(cs, min(bnd, int(pos[i]) - 1))
        rows.append((cs, bnd, cur, n, ps / n, i0, i - 1))
        cur, cs, ps, n, i0 = int(y_hat[i]), bnd + 1, float(p_int[i]), 1, i
    rows.append((cs, int(pos[-1]), cur, n, ps / n, i0, len(pos) - 1))
    df = pd.DataFrame(rows, columns=["start", "end", "state", "n_sites", "mean_p_int", "site_i0", "site_i1"])
    df["mean_p_int_bp"] = [_bp_weighted_posterior(int(r.start), int(r.end), int(r.site_i0), int(r.site_i1), pos, p_int)
                           for r in df.itertuples()]
    return df


def carve_unsupported(tr, pos, unsup):
    """Cut unsupported spans (1-based inclusive) out of the tracts as NA rows (state -1)."""
    if tr is None or len(tr) == 0 or unsup is None or len(unsup[0]) == 0:
        return tr
    m_s1, m_e1 = np.asarray(unsup[0], np.int64), np.asarray(unsup[1], np.int64)
    cols = list(tr.columns)
    keep_int = {"start", "end", "state", "n_sites", "site_i0", "site_i1"}

    def survive(base, a, b):
        d = dict(base)
        lo = int(np.searchsorted(pos, a, "left")); hi = int(np.searchsorted(pos, b, "right")) - 1
        d["start"], d["end"] = a, b
        d["n_sites"] = hi - lo + 1 if hi >= lo else 0
        d["site_i0"], d["site_i1"] = (lo, hi) if hi >= lo else (-1, -1)
        return d

    def nacall(base, a, b):
        d = {c: (np.nan if c not in keep_int else base.get(c)) for c in cols}
        d["start"], d["end"] = a, b
        d["state"], d["n_sites"] = -1, 0
        d["site_i0"], d["site_i1"] = -1, -1
        return d
    out = []
    for r in tr.to_dict("records"):
        s, e = int(r["start"]), int(r["end"])
        k = int(np.searchsorted(m_e1, s, "left")); cur = s
        while k < len(m_s1) and m_s1[k] <= e:
            ms, me = max(int(m_s1[k]), s), min(int(m_e1[k]), e)
            if ms > cur:
                out.append(survive(r, cur, ms - 1))
            out.append(nacall(r, ms, me))
            cur = me + 1; k += 1
        if cur <= e:
            out.append(survive(r, cur, e))
    return pd.DataFrame(out, columns=cols).sort_values("start").reset_index(drop=True)


def bp_spans(pos):
    """bp of each site's Voronoi cell (first / last cell end at the first / last site)."""
    pos = np.asarray(pos, float); n = len(pos)
    if n < 2:
        return np.ones(n)
    mid = 0.5 * (pos[:-1] + pos[1:])
    left = np.empty(n); right = np.empty(n)
    left[1:] = mid; right[:-1] = mid; left[0] = pos[0]; right[-1] = pos[-1]
    return np.maximum(right - left, 0.0)
