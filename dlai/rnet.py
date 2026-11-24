"""dlai rnet: genome-wide R_net structure test (dlaidraft2 "Structure diagnostic test"; port of
final_run/code/structure/rnet.py + rnet_gw.py, 2026-11-13).

One combination = one recipient + one sister + one donor genome (+ the outgroup), with that trio's own gene trees.
Per combination all chromosomes are laid end to end and the statistic is computed once for the genome:
  sites      outgroup homozygous REF and the three genomes called; R codes = recipient derived, sister homozygous
             ancestral (H01 H02 B00 B01 B02 of the 36-code layout, one sister genome: bins 0 / 1 (het) / 3 (hom))
  labels     the production label rule: T2 excess in supported bp, rank(f_d) + rank(donor-private)
  mirror     the donor-sister (T3) windows with recipient and sister swapped (12-pattern filter on the swapped genotypes)
  twin       the same rank rule on the mirrored T3 windows at the INT label's bp share q
  R_net = [JS*(INT, ILS) - JS*(twin top, twin rest)] / JS*(mirror, Background)
JS* = excess Jensen-Shannon divergence (log2, pseudocount 0.5) minus its mean over 100 multinomial draws from the
pooled spectrum. Genome value = mean over combinations. Jackknife: 20 equal-length blocks over the genome (common
coordinates for every combination; each replicate re-derives labels, q and twin); delete-one-chromosome jackknife with
several chromosomes; rnet_pool = the statistic of the counts summed over combinations.

  dlai rnet --vcf chr1.vcf.gz [chr2.vcf.gz ...] --gene-trees trees.tsv.gz [...] --combos combos.txt --outgroup og.txt
            --out PREFIX [--mask chr1.bed ...] [--workers N]
combos.txt: one combination per line, "recipient sister donor". Gene trees: per-trio calls (dlai/genetrees.py) holding
each combination's trio on every chromosome; one file with a chrom column, or one file per VCF in the same order.
-> PREFIX.tsv (one row: R_net, SE, CI, p05 = P(t_19 >= (R_net - 0.05) / SE), ...), PREFIX.combos.tsv (per combination)."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from .genetrees import read_gene_trees
from .labels import rank_windows
from .io import apply_mask, load_mask, load_vcf, read_samples, unsupported_overlap, unsupported_spans

NB, NNULL, DELTA, GAP = 20, 100, 0.05, 1_000_000
SH = ["ABB", "AHB", "ABH", "AHH", "HBB", "HBH"]
UN = ["AAB", "ABA", "BAB", "BBA", "HAB", "HBA"]
KEEP = SH + UN
NAMES = [l + p + s for l in "ABH" for p in "0123" for s in "012"]
MASK_R = np.isin(np.arange(36), [NAMES.index(c) for c in ["H01", "H02", "B00", "B01", "B02"]])
LET = lambda n: np.select([n == 0, n == 2, n == 1], ["A", "B", "H"], "?")
BIN1 = lambda n: np.select([n == 0, n == 1, n == 2], [0, 1, 3], -1)
BIN3 = lambda n: np.select([n == 0, n == 1, n == 2], [0, 1, 2], -1)
LV = lambda n: np.select([n == 0, n == 2, n == 1], [0, 1, 2], -1)


def code(r, p, d):
    """r = dosage of whoever plays recipient, p = plays sister, d = donor."""
    return LV(r) * 12 + BIN1(p) * 3 + BIN3(d)


def _js(p, q):
    m = (p + q) / 2
    kl = lambda x, y: float((x * np.log2(x / y)).sum())
    return (kl(p, m) + kl(q, m)) / 2


def jsx(a, b, rng, nnull=NNULL):
    mu = lambda c: (c + 0.5) / (c + 0.5).sum()
    keep = (a + b) > 0
    a, b = a[keep], b[keep]
    na, nb = int(round(a.sum())), int(round(b.sum()))
    if not na or not nb:
        return np.nan
    pool = (a + b) / (a + b).sum()
    return _js(mu(a), mu(b)) - np.mean([_js(mu(rng.multinomial(na, pool)), mu(rng.multinomial(nb, pool))) for _ in range(nnull)])


def label_sub(T, keep, target, fd, cdp, dpmin=2):
    """Production label rule restricted to the kept windows -> state per window (0 Other, 1 INT, 2 ILS, -1 none)."""
    t2i = T.t2i[keep[T.t2i]]
    ws = np.full(len(T.s), -1); ws[T.other & keep] = 0
    o, k, _ = rank_windows(fd[t2i], cdp[t2i], T.span[t2i], target)
    if k == 0 and len(o):
        k = 1
    ws[t2i] = 2; ws[t2i[o[:k]]] = 1
    d = (ws == 1) & (cdp < dpmin)
    if (ws == 1).sum() and d.sum() == (ws == 1).sum():
        d[t2i[o[0]]] = False
    ws[d] = 2
    return ws


def share_rule(T, idx, fd, cdp, q, dpmin=2):
    """The production rank over the window set idx, at bp share q: True = top."""
    if not len(idx):
        return np.zeros(0, bool)
    o, k, _ = rank_windows(fd[idx], cdp[idx], T.span[idx], q * T.span[idx].sum())
    k = max(k, 1)
    top = np.zeros(len(idx), bool); top[o[:k]] = True
    top &= ~(cdp[idx] < dpmin) | (idx == idx[o[0]])
    return top


# ------------------------------------------------------------------------------- one combination, one chromosome
def extract(sd, table, rc, sis, don, og, mask, gap_bp):
    """Window and site arrays of one combination on one chromosome."""
    n = lambda x: np.where(np.isfinite(sd.freq[x]), np.rint(2 * sd.freq[x]), -1).astype(int)
    n1, n2, n3, ng = n(sis), n(rc), n(don), n(og)
    s, e = table.start_pos.values.astype(np.int64), table.end_pos.values.astype(np.int64)
    t = table.topology.values.astype(str)
    span = (e - s + 1).astype(float)
    # supported bp: span minus the mask and the SNP-free stretches > gap_bp among the SNPs polymorphic in the four
    # genomes (the site table of a one-genome sub-trial)
    f4 = np.stack([sd.freq[x] for x in (sis, rc, don, og)]).astype(float)
    with np.errstate(invalid="ignore"):
        poly = (np.nanmax(np.where(np.isfinite(f4), f4, -1), 0) > 0) & (np.nanmin(np.where(np.isfinite(f4), f4, 2), 0) < 1)
    uns = unsupported_spans(sd.pos[poly], mask, gap_bp)
    if uns is not None:
        span = np.maximum(span - unsupported_overlap(s, e, uns), 0.0)
    ok = (n1 >= 0) & (n2 >= 0) & (n3 >= 0) & (ng == 0) & poly        # sites polymorphic in the four genomes
    pos, n1, n2, n3 = sd.pos[ok], n1[ok], n2[ok], n3[ok]
    pat = np.char.add(np.char.add(LET(n1), LET(n2)), LET(n3))
    mpa = np.char.add(np.char.add(LET(n2), LET(n1)), LET(n3))
    f_o, f_m = np.isin(pat, KEEP), np.isin(mpa, KEEP)
    cd, cm = code(n2, n1, n3), code(n1, n2, n3)
    w = np.searchsorted(s, pos, side="right") - 1
    ins = (w >= 0) & (w < len(s)) & (pos <= e[np.clip(w, 0, len(s) - 1)])
    w = np.where(ins, w, -1); m = w >= 0
    cb = lambda f: np.bincount(w[m & f], minlength=len(s)).astype(float)

    def fd_dp(p, dp):
        A, B, Aa, Ab = cb(np.isin(p, SH)), cb(p == "BAB"), cb(p == "AAB"), cb(p == "ABA")
        den = A + Ab + Aa
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, (A - B) / den, np.nan), cb(dp)
    fd0, cdp0 = fd_dp(pat, (n2 > 0) & (n1 == 0) & (n3 > 0))
    fdm, dpm = fd_dp(mpa, (n1 > 0) & (n2 == 0) & (n3 > 0))
    return dict(s=s, e=e, t=t, span=span, fd0=fd0, cdp0=cdp0, fdm=fdm, dpm=dpm, w=w[m].astype(np.int64), f_o=f_o[m],
                f_m=f_m[m], cd=cd[m], cm=cm[m], n_sites=int(m.sum()))


def concat(parts, offs):
    L = {k: [] for k in ("s", "e", "t", "span", "fd0", "cdp0", "fdm", "dpm", "w", "f_o", "f_m", "cd", "cm", "cw")}
    woff = 0
    for i, (p, off) in enumerate(zip(parts, offs)):
        L["s"].append(p["s"] + off); L["e"].append(p["e"] + off)
        for k in ("t", "span", "fd0", "cdp0", "fdm", "dpm", "f_o", "f_m", "cd", "cm"):
            L[k].append(p[k])
        L["w"].append(p["w"] + woff); L["cw"].append(np.full(len(p["s"]), i)); woff += len(p["s"])
    C = {k: np.concatenate(v) for k, v in L.items()}
    t = C["t"]
    G = SimpleNamespace(s=C["s"], e=C["e"], t=t, span=C["span"], t2i=np.flatnonzero(t == "T2"), other=np.isin(t, ["T1", "T3"]))
    return G, C


def stat(G, C, keep):
    """Labels, twin and the excess-JS terms on the kept windows."""
    count = lambda sel, c: np.bincount(c[sel], minlength=36).astype(float)[MASK_R]
    J = lambda a_, b_: jsx(a_, b_, np.random.default_rng(1), NNULL)
    isT2, isT3 = G.t == "T2", G.t == "T3"
    bpT2, bpT3 = G.span[isT2 & keep].sum(), G.span[isT3 & keep].sum()
    ws = label_sub(G, keep, max(bpT2 - bpT3, 0.0), C["fd0"], C["cdp0"])
    st = ws[C["w"]]
    q = G.span[ws == 1].sum() / bpT2 if bpT2 else np.nan
    t3i = np.flatnonzero(isT3 & keep)
    top = share_rule(G, t3i, C["fdm"], C["dpm"], q)
    hi, lo = np.zeros(len(G.s), bool), np.zeros(len(G.s), bool); hi[t3i[top]] = True; lo[t3i[~top]] = True
    k3 = np.zeros(len(G.s), bool); k3[t3i] = True
    f_o, f_m, cd, cm, w = C["f_o"], C["f_m"], C["cd"], C["cm"], C["w"]
    I_, L_, O_ = count(f_o & (st == 1), cd), count(f_o & (st == 2), cd), count(f_o & (st == 0), cd)
    M_, MH, ML = count(f_m & k3[w], cm), count(f_m & hi[w], cm), count(f_m & lo[w], cm)
    IL, HL, MO, IO = J(I_, L_), J(MH, ML), J(M_, O_), J(I_, O_)
    return dict(q=q, rnet=(IL - HL) / MO if MO > 0 else np.nan, R=IL / IO if IO > 0 else np.nan), np.vstack([I_, L_, O_, M_, MH, ML])


def jk(v, reps):
    reps = np.asarray(reps, float); nb = len(reps)
    se = float(np.sqrt((nb - 1) / nb * np.nansum((reps - np.nanmean(reps)) ** 2)))
    tq = tdist.ppf(0.975, nb - 1)
    return dict(se=se, lo=v - tq * se, hi=v + tq * se, p05=float(tdist.sf((v - DELTA) / se, nb - 1)) if se > 0 else np.nan,
                p0=float(tdist.sf(v / se, nb - 1)) if se > 0 else np.nan)


def rn_counts(c):
    J = lambda a_, b_: jsx(a_, b_, np.random.default_rng(1), NNULL)
    I, L, O, M, H, Lo = c
    mo = J(M, O)
    return (J(I, L) - J(H, Lo)) / mo if mo > 0 else np.nan


_P = {}


def _combo(args):
    """full statistic, 20 block replicates, delete-one-chromosome replicates and counts of one combination."""
    cb, chroms, offs, edges = args
    G, C = concat([_P[(cb, ch)] for ch in chroms], [offs[ch] for ch in chroms])
    full, cnt = stat(G, C, np.ones(len(G.s), bool))
    blk = np.clip(np.searchsorted(edges, G.s, side="right") - 1, 0, NB - 1)
    reps = [stat(G, C, blk != b) for b in range(NB)]
    rch = [stat(G, C, C["cw"] != i)[0]["rnet"] for i in range(len(chroms))] if len(chroms) > 1 else []
    return dict(full=full, cnt=cnt, r20=[x[0]["rnet"] for x in reps], c20=[x[1] for x in reps], rch=rch, n_win=len(G.s),
                n_sites=int(sum(_P[(cb, ch)]["n_sites"] for ch in chroms)))


def genome(P, chroms, combos, label="", workers=1):
    """P: {(combination, chrom): extract(...)} -> (genome row, per-combination table)."""
    _P.clear(); _P.update(P)
    offs, off = {}, 0
    for ch in chroms:                                   # common offsets: the same coordinates for every combination
        offs[ch] = off
        off += int(max(P[(cb, ch)]["e"].max() for cb in combos)) + GAP
    lo_ = min(int(P[(cb, ch)]["s"].min()) + offs[ch] for cb in combos for ch in chroms)
    hi_ = max(int(P[(cb, ch)]["e"].max()) + offs[ch] for cb in combos for ch in chroms)
    edges = np.linspace(lo_, hi_ + 1, NB + 1)
    jobs = [(cb, chroms, offs, edges) for cb in combos]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            res = list(ex.map(_combo, jobs))
    else:
        res = [_combo(j) for j in jobs]
    rows = [dict(label=label, combo="__".join(cb), **x["full"], n_win=x["n_win"], n_sites=x["n_sites"])
            for cb, x in zip(combos, res)]
    S = pd.DataFrame(rows)
    rn = float(np.nanmean(S.rnet))
    out = dict(label=label, n_combo=len(combos), n_chrom=len(chroms), rnet=rn,
               **jk(rn, np.nanmean(np.array([x["r20"] for x in res]), axis=0)),
               R_mean=float(np.nanmean(S.R)), q_mean=float(np.nanmean(S.q)))
    if len(chroms) > 1:
        out.update({f"chr_{k}": v for k, v in jk(rn, np.nanmean(np.array([x["rch"] for x in res]), axis=0)).items()})
    out["rnet_pool"] = rn_counts(sum(x["cnt"] for x in res))
    out.update({f"pool_{k}": v for k, v in jk(out["rnet_pool"],
                                               [rn_counts(sum(x["c20"][b] for x in res)) for b in range(NB)]).items()})
    return out, S


# -------------------------------------------------------------------------------------------------------- CLI
def add_parser(sp):
    p = sp.add_parser("rnet", help="genome-wide R_net structure test", description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf", nargs="+", required=True,
                   help="one VCF per chromosome, or one VCF holding several contigs together with --contigs")
    p.add_argument("--contigs", help="comma-separated contigs to read from a single multi-contig --vcf (indexed: read "
                                     "through the index)")
    p.add_argument("--gene-trees", nargs="+", required=True,
                   help="per-trio gene-tree calls: one file with a chrom column, or one file per VCF")
    p.add_argument("--combos", required=True, help="file: recipient sister donor per line")
    p.add_argument("--outgroup", required=True, help="outgroup sample file (first name used)")
    p.add_argument("--mask", nargs="+", help="exclusion BED(s): one for all contigs, or one per VCF")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--label", default="", help="label written in the label column of both outputs")
    p.add_argument("--max-site-gap", type=int, default=100000,
                   help="SNP-free stretches longer than this (bp) do not count as supported window bp (default 100000)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                   help="combinations computed in parallel (default $SLURM_CPUS_PER_TASK or 1)")
    return p


def run(a):
    og = read_samples(a.outgroup)[0]
    combos = []
    for ln in open(a.combos):
        f = ln.split()
        if f and not f[0].startswith("#"):
            if len(f) != 3:
                raise SystemExit(f"{a.combos}: expected 'recipient sister donor', got {ln.strip()!r}")
            combos.append(tuple(f))
    samples = sorted({x for c in combos for x in c} | {og})
    if a.contigs and len(a.vcf) != 1:
        raise SystemExit("--contigs needs exactly one --vcf")
    reads = [(a.vcf[0], c) for c in a.contigs.split(",")] if a.contigs else [(v, None) for v in a.vcf]
    if len(a.gene_trees) not in (1, len(reads)) or (a.mask and len(a.mask) not in (1, len(reads))):
        raise SystemExit("--gene-trees / --mask: one file, or one per chromosome")
    P, chroms = {}, []
    for i, (vcf, contig) in enumerate(reads):
        sd = load_vcf(vcf, samples, contig)
        mask = None
        if a.mask:
            mask = load_mask(a.mask[min(i, len(a.mask) - 1)], sd.chrom)
            sd = apply_mask(sd, mask)
        gtf = a.gene_trees[min(i, len(a.gene_trees) - 1)]
        chroms.append(sd.chrom)
        for rc, sis, don in combos:
            t = read_gene_trees(gtf, [rc], [sis], [don], chrom=sd.chrom if len(a.gene_trees) == 1 else None,
                                log=lambda *x: None, trio=(sis, don))[rc]
            P[((rc, sis, don), sd.chrom)] = extract(sd, t, rc, sis, don, og, mask, a.max_site_gap)
        print(f"[rnet] {sd.chrom}: {len(combos)} combination(s) extracted", flush=True)
    out, S = genome(P, chroms, combos, a.label, a.workers)
    pd.DataFrame([out]).to_csv(f"{a.out}.tsv", sep="\t", index=False)
    S.to_csv(f"{a.out}.combos.tsv", sep="\t", index=False)
    print(pd.DataFrame([out]).T.to_string())
    return 0
