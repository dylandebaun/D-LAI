"""Gene-tree input.

dlai does not infer gene trees; they are supplied by the user (core_final/genetrees/ holds a sticcs and an IQ-TREE
builder that write this format). Two forms, both a tab-separated file (optionally gzipped) with a header line and
1-based inclusive coordinates:

  per-trio calls   chrom  start  end  recipient  sister  donor  topology
                   one row per local-tree interval of one (sister, recipient, donor) trio. dlai votes them into one
                   table per recipient (vote_trios): the trios' intervals are stacked on the union of their
                   breakpoints, each segment takes the plurality topology over trios, a tie takes the reference
                   trio's call (first --p1 sample x first --p3 sample) when it is among the tied topologies and is
                   otherwise drawn at random (seeded per recipient), and consecutive segments with the same call are
                   merged into one window. This is the rule of the sticcs builder used for the production runs.
  voted tables     chrom  start  end  recipient  topology
                   one window per row, already voted per recipient; used as given (consecutive windows with the
                   same call are still merged unless keep_windows).

topology is T1 = ((sister, recipient), donor), T2 = ((recipient, donor), sister), T3 = ((sister, donor), recipient),
rooted with the outgroup; NA or an empty field = unresolved. The chrom column may be left out for a single-contig run.

A directory is also accepted: the per-recipient tables written by the final_run/core builders
(dlai_<recipient>_gt_topo_<p1>_<p3>_w25.tsv with start_pos, end_pos, topology[, n_snps, w_T*, med_d*]), read as voted
tables.

Output of read_gene_trees: {recipient: DataFrame(window_id, start_pos, end_pos, topology[, w_T1, w_T2, w_T3,
n_snps, med_d12, med_d23, med_d13, med_d2o])}.
"""
import functools
import glob
import os
import re

import numpy as np
import pandas as pd

TOPOS = np.array(["T1", "T2", "T3"])
_COLS_TRIO = ["start", "end", "recipient", "sister", "donor", "topology"]
_COLS_VOTED = ["start", "end", "recipient", "topology"]


def _norm_topology(s):
    t = s.fillna("NA").astype(str).str.strip()
    t = t.where(t.isin(["T1", "T2", "T3"]), "NA")
    return t.values.astype(object)


def plurality_rows(w, rng, ref=None):
    """Row-wise plurality of an (n, 3) vote matrix (columns T1, T2, T3); an all-zero row -> NA. A tie takes ref[i]
    (a topology string) when it is among the tied maxima, otherwise a random pick among them; the random picks are
    drawn one at a time in row order (the generator stream of the sticcs builder)."""
    out = np.full(len(w), "NA", object)
    mx = w.max(1)
    hit = w == mx[:, None]
    nt = hit.sum(1)
    one = (mx > 0) & (nt == 1)
    out[one] = TOPOS[hit[one].argmax(1)]
    tied = np.flatnonzero((mx > 0) & (nt > 1))
    if ref is not None and len(tied):
        rt = np.asarray(ref, object)[tied]
        code = np.full(len(tied), -1)
        for j, t in enumerate(TOPOS):
            code[rt == t] = j
        use = code >= 0
        use[use] = hit[tied[use], code[use]]
        out[tied[use]] = rt[use]
        tied = tied[~use]
    for i in tied:
        out[i] = TOPOS[rng.choice(np.flatnonzero(hit[i]))]
    return out


def merge_windows(t):
    """Collapse consecutive windows with the same topology (NA runs included) into one window."""
    grp = (t.topology != t.topology.shift()).cumsum()
    agg = dict(start_pos=("start_pos", "first"), end_pos=("end_pos", "last"), topology=("topology", "first"))
    for c in ("w_T1", "w_T2", "w_T3"):
        if c in t:
            agg[c] = (c, "mean")
    return t.groupby(grp).agg(**agg).reset_index(drop=True)


def vote_trios(d, reference=None, seed=1):
    """Per-trio calls of ONE recipient -> its voted table (start_pos, end_pos, topology, w_T1..3), unmerged.
    d: DataFrame(start, end, sister, donor, topology); reference: (sister, donor) of the reference trio."""
    d = d.sort_values(["sister", "donor", "start"], kind="mergesort")
    trios = list(dict.fromkeys(zip(d.sister.values, d.donor.values)))
    if reference in trios:                                  # the reference trio first (its call breaks ties)
        trios.remove(reference); trios.insert(0, reference)
    else:
        reference = None
    st_all, en_all = d.start.values.astype(np.int64), d.end.values.astype(np.int64)
    last = int(en_all.max())
    grid = np.unique(np.r_[st_all, en_all[en_all < last] + 1])
    gend = np.r_[grid[1:] - 1, last]
    votes = np.zeros((len(grid), 3))
    ref_call = None
    by ={k: g for k, g in d.groupby(["sister", "donor"], sort=False)}
    for k in trios:
        g = by[k]
        st, en, tp = g.start.values, g.end.values, g.topology.values
        i = np.searchsorted(st, grid, side="right") - 1
        ok = i >= 0
        ok[ok] = en[i[ok]] >= grid[ok]                     # segment covered by one of this trio's intervals
        call = np.full(len(grid), "NA", object)
        call[ok] = tp[i[ok]]
        if ref_call is None and reference is not None:
            ref_call = call
        for j, t in enumerate(TOPOS):
            votes[:, j] += call == t
    rng = np.random.default_rng(seed)
    t = pd.DataFrame(dict(start_pos=grid, end_pos=gend, topology=plurality_rows(votes, rng, ref_call)))
    sh = votes / np.maximum(votes.sum(1, keepdims=True), 1e-12)
    t["w_T1"], t["w_T2"], t["w_T3"] = sh[:, 0], sh[:, 1], sh[:, 2]
    return t, len(trios)


@functools.lru_cache(maxsize=4)
def _read_table(path):
    """the gene-tree file, topology normalised (cached: dlai rnet reads one trio at a time from the same file)."""
    d = pd.read_csv(path, sep="\t", dtype={"chrom": str, "recipient": str, "sister": str, "donor": str,
                                           "topology": str}, keep_default_na=False, na_values=[""])
    d["topology"] = _norm_topology(d.topology)
    return d


def read_gene_trees(path, recipients, p1, p3, chrom=None, seed=1, keep_windows=False, log=print, trio=None):
    """Read the gene-tree input for `recipients` (see the module docstring) -> {recipient: window table}. trio =
    (sister, donor): per-trio calls of that one trio only (dlai rnet)."""
    out = {}
    if os.path.isdir(path):
        for r in recipients:
            f = sorted(glob.glob(os.path.join(path, f"dlai_{r}_gt_topo_*.tsv")))
            f = [x for x in f if re.search(rf"dlai_{re.escape(r)}_gt_topo_[^/]*_w\d+\.tsv$", x)]
            if not f:
                raise SystemExit(f"gene trees: no dlai_{r}_gt_topo_*.tsv table in {path}")
            t = pd.read_csv(f[0], sep="\t")
            t["topology"] = _norm_topology(t.topology)
            out[r] = t
        log(f"gene trees: {len(out)} per-recipient table(s) from {path}")
    else:
        d = _read_table(path)
        miss = [c for c in _COLS_VOTED if c not in d.columns]
        if miss:
            raise SystemExit(f"gene trees {path}: missing column(s) {', '.join(miss)} "
                             f"(need {' '.join(_COLS_VOTED)} [+ sister donor for per-trio calls])")
        if "chrom" in d.columns:
            if chrom is None:
                cs = d.chrom.unique()
                if len(cs) > 1:
                    raise SystemExit(f"gene trees {path}: {len(cs)} contigs and no contig given")
            else:
                d = d[d.chrom == str(chrom)]
        trio_fmt = "sister" in d.columns and "donor" in d.columns
        if trio_fmt != ("sister" in d.columns or "donor" in d.columns):
            raise SystemExit(f"gene trees {path}: per-trio calls need BOTH sister and donor columns")
        ref = (p1[0], p3[0])
        for r in recipients:
            g = d[d.recipient == r]
            if trio is not None and trio_fmt:
                g = g[(g.sister == trio[0]) & (g.donor == trio[1])]
            if g.empty:
                raise SystemExit(f"gene trees {path}: no rows for recipient {r}" + (f" on {chrom}" if chrom else "")
                                 + (f" (trio {trio[0]} x {trio[1]})" if trio else ""))
            if trio_fmt:
                t, n = vote_trios(g[["start", "end", "sister", "donor", "topology"]], ref, seed)
                log(f"gene trees: {r}: {n} trio(s) voted"
                    + ("" if ref in set(zip(g.sister, g.donor)) else f" (no reference trio {ref[0]} x {ref[1]}: "
                                                                       f"ties drawn at random)"))
            else:
                g = g.sort_values("start", kind="mergesort")
                t = pd.DataFrame(dict(start_pos=g.start.values.astype(np.int64),
                                      end_pos=g.end.values.astype(np.int64), topology=g.topology.values))
                if (t.start_pos.values[1:] <= t.end_pos.values[:-1]).any():
                    raise SystemExit(f"gene trees {path}: overlapping windows for recipient {r} (voted tables need "
                                     f"one call per position; give per-trio calls with sister/donor columns to vote)")
            out[r] = t
        log(f"gene trees: {len(out)} recipient(s) from {path} ({'per-trio calls, voted' if trio_fmt else 'voted tables'})")
    for r, t in out.items():
        if not keep_windows:
            m = merge_windows(t)
            if len(m) != len(t):                # n_snps / med_d* of the input describe the unmerged windows: dropped
                out[r] = m
        out[r] = out[r].reset_index(drop=True)
        if "window_id" in out[r]:
            out[r] = out[r].drop(columns="window_id")
        out[r].insert(0, "window_id", np.arange(len(out[r])))
    return out


def window_distances(t, sd, keep, r, p1, p3, og):
    """Per-window genotype distances (diagnostics only; the formula of the sticcs builder): over the window's
    outgroup-REF sites, the mean absolute derived-fraction difference between the recipient and the P1 panel mean (d12),
    the P3 panel mean (d23), the outgroup (d2o), and between the two panel means (d13); uncalled -> no contribution.
    Adds n_snps and med_d12 / med_d23 / med_d13 / med_d2o to t (in place) unless t already has them."""
    if {"med_d12", "med_d23", "med_d13", "med_d2o"} <= set(t.columns):
        return t
    pos = sd.pos[keep]
    with np.errstate(invalid="ignore"):
        f = lambda s: sd.freq[s][keep].astype(float)
        P1m = np.nanmean(np.stack([f(s) for s in p1], 1), 1); P3m = np.nanmean(np.stack([f(s) for s in p3], 1), 1)
    x = f(r)
    csum = lambda v: np.r_[0.0, np.cumsum(np.nan_to_num(v))]
    i0 = np.searchsorted(pos, t.start_pos.values); i1 = np.searchsorted(pos, t.end_pos.values, side="right")
    n = np.maximum(i1 - i0, 1)
    t["n_snps"] = i1 - i0
    for nm, v in (("d12", np.abs(x - P1m)), ("d23", np.abs(x - P3m)), ("d13", np.abs(P1m - P3m)), ("d2o", np.abs(x - f(og)))):
        c = csum(v); t["med_" + nm] = (c[i1] - c[i0]) / n
    return t
