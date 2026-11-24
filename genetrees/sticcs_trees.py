#!/usr/bin/env python
"""sticcs_trees.py -- gene-tree input for dlai from stacked sticcs sub-ARGs (separate from dlai; dlai never runs it).

Method (Martin 2025, Genetics iyaf181; twisst2 "sticcstack"): for each recipient, K (sister, donor) pairs -- the
reference pair (first --p1 x first --p3 sample) first, then the other pairs in one seeded shuffle, the first K used --
each get their own sticcs inference on the 3 diploids (sister, recipient, donor; sites with 2..5 derived copies among
their 6 leaves, outgroup homozygous REF). In every sub-ARG tree each recipient haplotype gets the rooted-triplet
plurality over the 2 x 2 (sister leaf, donor leaf) pairs (a tie takes the reference leaf pair's triplet when it is among
the tied ones, else a seeded random pick), and the diploid call is "any": T2/T3 if either haplotype is discordant (T2 vs
T3: the haplotype with the larger winning share, equal -> the first), else T1 if either haplotype is T1, else NA.

Output (--out, gzipped TSV, 1-based inclusive coordinates; the gene-tree input format of dlai, see dlai/genetrees.py):
  default   per-trio calls  chrom start end recipient sister donor topology   (one row per sub-ARG tree; dlai votes)
  --vote    voted tables    chrom start end recipient topology w_T1 w_T2 w_T3  (the vote done here: the production
            final_run/core/sticcs_stack_gene_trees.py --variants vote_merge --tie ref --pair-order nested; the same
            windows and calls, without its diagnostic columns n_snps / med_d*, which dlai computes itself). Use it when
            the per-trio file would be too large (recipients x K x trees).

    python sticcs_trees.py --vcf X.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt --outgroup og.txt \\
        --out trees.tsv.gz [--combos 64] [--vote] [--n-workers N] [--phased]
Sample files: whitespace-separated names. One contig per VCF (biallelic SNPs; ancestral = REF, outgroup hom-REF sites).
"""
import argparse
import gzip
import itertools
import os
import random
import sys
import time
import types
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "external", "sticcs"))
sys.path.insert(0, os.path.dirname(HERE))                    # core_final: the plurality rule is dlai's own
sys.modules.setdefault("cyvcf2", types.ModuleType("cyvcf2"))   # sticcs imports it for its own VCF reader only
from sticcs.sticcs import get_patterns_and_matches, get_clusters, infer_trees   # noqa: E402
from dlai.genetrees import TOPOS, plurality_rows   # noqa: E402
_G = {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, h in (("vcf", "VCF (one contig; biallelic SNPs used)"), ("p1", "sister (P1) panel sample file"),
                 ("p2", "recipient (P2) sample file"), ("p3", "donor (P3) panel sample file"),
                 ("outgroup", "outgroup sample file (first name used; ancestral = REF)"),
                 ("out", "output TSV (.gz)")):
        p.add_argument("--" + k, required=True, help=h)
    p.add_argument("--combos", type=int, default=64, help="(sister, donor) pairs per recipient (default 64)")
    p.add_argument("--vote", action="store_true", help="write voted per-recipient tables instead of per-trio calls")
    p.add_argument("--seq-len", type=int, default=None, help="end of the last tree (default: last SNP position)")
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                   help="worker processes (default $SLURM_CPUS_PER_TASK or 4)")
    p.add_argument("--seed", type=int, default=1, help="pair-order and tie seed (default 1)")
    p.add_argument("--phased", action="store_true",
                   help="use the VCF's phase: every diploid becomes two haploid sticcs samples (the VCF must be phased)")
    return p.parse_args()


def read_samples(path):
    return open(path).read().split()


def load_genotypes(vcf, samples):
    import allel
    v = allel.read_vcf(vcf, fields=["samples", "variants/CHROM", "variants/POS", "calldata/GT"], alt_number=1)
    ch = np.unique(v["variants/CHROM"])
    if len(ch) > 1:
        sys.exit(f"{vcf} holds {len(ch)} contigs ({', '.join(map(str, ch[:3]))}, ...); run one contig at a time")
    sm = list(v["samples"])
    miss = [s for s in samples if s not in sm]
    if miss:
        sys.exit(f"samples not in VCF: {miss[:5]}")
    return str(ch[0]), np.asarray(v["variants/POS"], np.int64), allel.GenotypeArray(v["calldata/GT"]), sm


# ---- rooted-triplet weights of one recipient haplotype against every (sister leaf, donor leaf) pair, in one tree
def leaf_sets(tree):
    """node -> array of the leaf ids below it (sticcs Tree: leaves 0..n-1, node_children dict)."""
    out, stack = {}, [(tree.root, False)]
    while stack:
        nd, done = stack.pop()
        ch = tree.node_children.get(nd, [])
        if not ch:
            out[nd] = np.array([nd], np.int32)
        elif done:
            out[nd] = np.concatenate([out[c] for c in ch])
        else:
            stack.append((nd, True)); stack.extend((c, False) for c in ch)
    return out


def hap_levels(tree, h, leaves, n_leaves):
    """level[x] = index k of the first ancestor of leaf h (h itself = 0) that contains leaf x; sub[x] = the child
    subtree of that ancestor holding x. Two leaves with equal level and sub form a clade that excludes h."""
    level = np.full(n_leaves, -1, np.int32); sub = np.full(n_leaves, -1, np.int64)
    prev, node, k = h, tree.node_parent.get(h), 1
    level[h] = 0
    while node is not None:
        for c in tree.node_children[node]:
            if c != prev:
                ls = leaves[c]; level[ls] = k; sub[ls] = c
        prev, node, k = node, tree.node_parent.get(node), k + 1
    return level, sub


def triplet_weights(level, sub, p1_leaves, p3_leaves):
    """counts of (T1, T2, T3) over all sister x donor leaf pairs; unresolved pairs are not counted."""
    l1, l3 = level[p1_leaves][:, None], level[p3_leaves][None, :]
    t1 = int((l1 < l3).sum()); t2 = int((l3 < l1).sum())
    t3 = int(((l1 == l3) & (sub[p1_leaves][:, None] == sub[p3_leaves][None, :])).sum())
    return np.array([t1, t2, t3], float)


def one_combo(job):
    """sticcs on (sister, recipient, donor) -> (starts, ends, wa, wb, ra, rb), each (n trees, 3): triplet counts over
    the 2 x 2 leaf pairs per recipient haplotype (wa, wb) and over the reference leaf pair only (ra, rb)."""
    r, i1, i3 = job
    pos, seq_len = _G["pos"], _G["seq_len"]
    ind = (i1, r, i3)
    p1l, hl, p3l = np.array([0, 1]), (2, 3), np.array([4, 5])
    if _G.get("hap") is not None:
        der = _G["hap"][:, [c for i in ind for c in (2 * i, 2 * i + 1)]]
        pl = np.ones(6, int)
    else:
        der = _G["dc"][:, ind]
        pl = np.array([2, 2, 2])
    s = der.sum(1)
    ok = (der >= 0).all(1) & (s > 1) & (s < 6)
    der, sp = der[ok], pos[ok]
    if len(sp) == 0:
        return job, None
    pat, mat, _ = get_patterns_and_matches(der)
    cl = get_clusters(pat, mat, sp, ploidies=pl, second_chances=True, seq_start=1, seq_len=seq_len, silent=True)
    st, en, wa, wb, ra, rb = [], [], [], [], [], []
    for tree in infer_trees(pat, pl, cl, multi_pass=True, silent=True):
        lv = leaf_sets(tree)
        st.append(int(tree.interval[0])); en.append(int(tree.interval[1]))
        la, lb = hap_levels(tree, hl[0], lv, 6), hap_levels(tree, hl[1], lv, 6)
        wa.append(triplet_weights(*la, p1l, p3l)); wb.append(triplet_weights(*lb, p1l, p3l))
        ra.append(triplet_weights(*la, p1l[:1], p3l[:1])); rb.append(triplet_weights(*lb, p1l[:1], p3l[:1]))
    return job, tuple(np.array(x) for x in (st, en, wa, wb, ra, rb))


def run_chunk(chunk):
    return [one_combo(j) for j in chunk]


def stream_runs(ex, jobs, chunk, ahead):
    """(job, result) in job order with at most `ahead` chunks in flight, so finished sub-ARGs cannot pile up."""
    it = iter([jobs[i:i + chunk] for i in range(0, len(jobs), chunk)])
    q = deque(ex.submit(run_chunk, c) for c in itertools.islice(it, ahead))
    while q:
        out = q.popleft().result()
        c = next(it, None)
        if c is not None:
            q.append(ex.submit(run_chunk, c))
        yield from out


def any_call_rows(wa, wb, rng, ra, rb):
    """Diploid "any" call per tree (module docstring), haplotype ties broken by the reference leaf pair."""
    ta = plurality_rows(wa, rng, plurality_rows(ra, rng)); tb = plurality_rows(wb, rng, plurality_rows(rb, rng))
    sa = wa.max(1) / np.maximum(wa.sum(1), 1e-12); sb = wb.max(1) / np.maximum(wb.sum(1), 1e-12)
    out = np.full(len(ta), "NA", object)
    da = (ta == "T2") | (ta == "T3"); db = (tb == "T2") | (tb == "T3")
    out[(ta == "T1") | (tb == "T1")] = "T1"
    out[db] = tb[db]
    out[da] = ta[da]
    for i in np.flatnonzero(da & db & (ta != tb)):
        out[i] = ta[i] if sa[i] >= sb[i] else tb[i]
    return out


def recipient_rows(r, subs, names, seq_len, seed, vote):
    """One recipient's output rows. subs: its sub-ARGs in pair order (the reference pair first), names: the matching
    (sister, donor) names. The generator is consumed exactly as by the production builder, so --vote reproduces its
    tables and the per-trio calls are its per-pair calls."""
    rng = np.random.default_rng(seed)
    grid = np.unique(np.concatenate([s[0] for s in subs]))              # union of breakpoints (tree starts)
    gend = np.r_[grid[1:] - 1, seq_len]
    votes = np.zeros((len(grid), 3)); ref_v = None; rows = []
    for (st, en, wa, wb, ra, rb), (sis, don) in zip(subs, names):
        tree_call = any_call_rows(wa, wb, rng, ra, rb)
        k = np.clip(np.searchsorted(st, grid, side="right") - 1, 0, len(st) - 1)
        call = tree_call[k]
        if ref_v is None:
            ref_v = call
            plurality_rows((wa + wb)[k], rng)    # the production builder's stacked-count reference: no output, but it
                                                 # draws from the generator, so it stays for identical tables
        if vote:
            for j, t in enumerate(TOPOS):
                votes[:, j] += call == t
        else:
            rows.append((st, en, sis, don, tree_call))
    if not vote:
        return rows
    topo = plurality_rows(votes, rng, ref_v)
    sh = votes / np.maximum(votes.sum(1, keepdims=True), 1e-12)
    b = np.flatnonzero(np.r_[True, topo[1:] != topo[:-1]])               # merge runs of the same call
    e = np.r_[b[1:], len(topo)]
    w = np.stack([np.add.reduceat(sh[:, j], b) / (e - b) for j in range(3)], 1)
    return grid[b], gend[e - 1], topo[b], w


def main():
    a = parse_args()
    p1, p3 = read_samples(a.p1), read_samples(a.p3)
    og = read_samples(a.outgroup)[0]
    p2 = read_samples(a.p2)
    inds = list(dict.fromkeys(p1 + p2 + p3))
    t0 = time.time()
    chrom, pos, G, sm = load_genotypes(a.vcf, inds + [og])
    col = {s: sm.index(s) for s in inds + [og]}
    og_ref = G[:, col[og]].to_n_alt(fill=-1) == 0
    dc = G[:, [col[s] for s in inds]].to_n_alt(fill=-1)
    seq_len = a.seq_len or int(pos.max())
    _G.update(dc=dc[og_ref], pos=pos[og_ref], seq_len=seq_len, hap=None)
    if a.phased:
        n_unph = 0                      # phase check on the raw text (scikit-allel's is_phased is unreliable here)
        with (gzip.open(a.vcf, "rt") if a.vcf.endswith(".gz") else open(a.vcf)) as fh:
            for ln in fh:
                if ln[0] != "#":
                    n_unph += sum("/" in f.split(":", 1)[0] for f in ln.rstrip("\n").split("\t")[9:])
        if n_unph:
            sys.exit(f"--phased: {n_unph:,} unphased ('/') genotypes in {a.vcf}")
        gt = np.asarray(G)[:, [col[s] for s in inds]]
        _G["hap"] = gt[og_ref].reshape(int(og_ref.sum()), -1).astype(np.int8)
    ix = {s: i for i, s in enumerate(inds)}
    pairs = [(x, y) for x in p1 for y in p3]
    rest = pairs[1:]; random.Random(a.seed).shuffle(rest)
    sel = ([pairs[0]] + rest)[:a.combos]
    jobs = [(ix[r], ix[x], ix[y]) for r in p2 for x, y in sel]
    print(f"[sticcs] {og_ref.sum():,} outgroup-REF sites; {len(p2)} recipients x {len(sel)} (sister, donor) pairs = "
          f"{len(jobs)} sticcs runs on {a.n_workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    n_rows, ntr, cur, q = 0, [], [], 0
    with gzip.open(tmp, "wt") as fh, ProcessPoolExecutor(a.n_workers) as ex:
        fh.write("chrom\tstart\tend\trecipient\t" + ("topology\tw_T1\tw_T2\tw_T3\n" if a.vote
                                                     else "sister\tdonor\ttopology\n"))
        for k, (job, out) in enumerate(stream_runs(ex, jobs, 4, 4 * a.n_workers)):
            if out is not None:
                ntr.append(len(out[0]))
            cur.append((out, sel[k % len(sel)]))
            if (k + 1) % max(1, len(jobs) // 20) == 0:
                print(f"[sticcs]   {k + 1}/{len(jobs)} runs ({time.time() - t0:.0f}s)", flush=True)
            if len(cur) == len(sel):          # all of this recipient's runs are in: write it, free it
                r = p2[q]
                assert job[0] == ix[r], (job, r)
                ok = [(o, nm) for o, nm in cur if o is not None]
                if not ok:
                    print(f"[sticcs] {r}: no informative sites in any pair, no rows", flush=True)
                else:
                    res = recipient_rows(r, [o for o, _ in ok], [nm for _, nm in ok], seq_len, a.seed, a.vote)
                    if a.vote:
                        st, en, tp, w = res
                        for i in range(len(st)):
                            fh.write(f"{chrom}\t{st[i]}\t{en[i]}\t{r}\t{tp[i]}\t{float(w[i, 0])!r}\t{float(w[i, 1])!r}\t{float(w[i, 2])!r}\n")
                        n_rows += len(st)
                    else:
                        for st, en, sis, don, call in res:
                            fh.writelines(f"{chrom}\t{s}\t{e}\t{r}\t{sis}\t{don}\t{c}\n" for s, e, c in zip(st, en, call))
                            n_rows += len(st)
                cur, q = [], q + 1
    os.replace(tmp, a.out)
    print(f"[sticcs] done in {time.time() - t0:.0f}s: {n_rows:,} rows -> {a.out}; trees per sub-ARG median "
          f"{np.median(ntr) if ntr else 0:.0f}", flush=True)


if __name__ == "__main__":
    main()
