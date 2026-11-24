#!/usr/bin/env python
"""iqtree_trees.py -- gene-tree input for dlai from per-window 4-taxon IQ-TREE trees (separate from dlai; dlai never
runs it).

Method (the production IQ-TREE gene trees: final_run/core/dlai_v2_core.py estimate_gene_trees_multi, all-combos path):
sites are the biallelic A/C/G/T SNPs of one contig (records sharing a position dropped) where the outgroup (first
--outgroup sample) is called homozygous REF. Each recipient gets its own window grid: non-overlapping --snp-window (25)
SNP windows over the sites called in its reference quartet (first --p1, recipient, first --p3, outgroup), from the first
such site on; a trailing partial window is not used, and a window spans its first to its last SNP (consecutive windows
do not abut). K (sister, donor) pairs from --p1 x --p3, the same K for every window and recipient: --pair-order random =
random.Random(seed).sample(all pairs, K) (every pair, in --p1 x --p3 order, when K >= their number); nested = the
reference pair (first --p1 x first --p3) first, then the other pairs in one random.Random(seed).shuffle, the first K used
(GT_PAIR_ORDER=nested). Per window and pair, the 4-taxon alignment (sister, recipient, donor, outgroup) of the window's
SNPs, as IUPAC genotypes (heterozygote -> ambiguity code, uncalled -> N; the non-reference sister/donor may be uncalled),
gets an ML tree from IQ-TREE (-m GTR --fast -T 1 -seed); alignments are run --batch-size at a time through one IQ-TREE
-S call, the batches in parallel. The unrooted quartet, rooted with the outgroup, is the call; a failed or unresolved
tree is NA.

Output (--out, gzipped TSV, 1-based inclusive coordinates; the per-trio gene-tree input of dlai, see dlai/genetrees.py):
  chrom start end recipient sister donor topology   one row per window per pair: recipients in --p2 order, windows in
                                                    order, and within a window the pairs in pair order.
The production per-window table (dlai_<recipient>_gt_topo_*_w25.tsv) is the plurality over a window's rows in file
order, NA not counted, a tie going to the topology encountered first, NA if none resolved (_majority_vote_topo):
    def vote(t):
        c = collections.Counter(x for x in t if x in ("T1", "T2", "T3"))
        return c.most_common(1)[0][0] if c else "NA"
    d = pd.read_csv(out, sep="\\t", keep_default_na=False)
    d.groupby(["recipient", "start", "end"], sort=False).topology.agg(vote)
(dlai votes the rows itself, with its own tie rule.) Identical trees need the production batch layout as well as its
inputs: IQ-TREE's tree for one alignment depends on the other alignments in its -S call, so recipients (--p2 order),
pairs, --snp-window, --batch-size, --iqtree-seed and the IQ-TREE binary must all match; the alignment files carry the
production names (<recipient>__<window>__<sister>__<donor>.fasta, which -S processes in sorted order).

    python iqtree_trees.py --vcf X.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt --outgroup og.txt \\
        --out trees.tsv.gz [--combos 5] [--pair-order random|nested] [--seed 2] [--n-workers N]
Sample files: whitespace-separated names. One contig per VCF. Defaults are the production ones (dlai_v2.py --gene-trees
iqtree: 5 pairs, random order, pair seed 2, IQ-TREE seed 1, 25-SNP windows, 5,000 alignments per IQ-TREE call).
Decode these trees with `dlai decode --keep-windows`: the IQ-TREE path of the old core used its 25-SNP windows as given
(never merged). The gaps between windows become NA segments in dlai's vote, which carry no label, exactly like the
positions outside every window in the old core.
"""
import argparse
import array
import gzip
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

IQTREE = "/home/ddebaun/nas5/miniconda3/envs/introg/bin/iqtree3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, h in (("vcf", "VCF (one contig; biallelic SNPs used)"), ("p1", "sister (P1) panel sample file"),
                 ("p2", "recipient (P2) sample file"), ("p3", "donor (P3) panel sample file"),
                 ("outgroup", "outgroup sample file (first name used; ancestral = REF)"),
                 ("out", "output TSV (.gz)")):
        p.add_argument("--" + k, required=True, help=h)
    p.add_argument("--combos", type=int, default=5,
                   help="(sister, donor) pairs per window (default 5; >= the number of pairs = every pair)")
    p.add_argument("--pair-order", choices=["random", "nested"], default="random",
                   help="random: a seeded sample of --combos pairs; nested: the reference pair first, then a seeded "
                        "shuffle, the first --combos used (default random)")
    p.add_argument("--seed", type=int, default=2, help="pair-selection seed (production gt_combos_seed, default 2)")
    p.add_argument("--iqtree-seed", type=int, default=1, help="IQ-TREE -seed (production --seed, default 1)")
    p.add_argument("--snp-window", type=int, default=25, help="SNPs per window (default 25)")
    p.add_argument("--batch-size", type=int, default=5000, help="alignments per IQ-TREE -S call (default 5000)")
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                   help="worker processes (default $SLURM_CPUS_PER_TASK or 4)")
    p.add_argument("--iqtree-bin", default=IQTREE, help=f"IQ-TREE 2/3 binary (default {IQTREE})")
    return p.parse_args()


def read_samples(path):
    return open(path).read().split()


# ---- sites (final_run/core load_site_data, no mask)
def _alt_fraction(gt_field):
    """VCF GT field -> ALT-allele fraction (0, 0.5, 1 for diploids), NaN if any allele is missing."""
    gt = gt_field.split(":", 1)[0]
    alleles = gt.split("|" if "|" in gt else "/")
    d = 0
    for a in alleles:
        if a == ".":
            return np.nan
        if a != "0":
            d += 1
    return d / len(alleles)


def load_sites(vcf, samples):
    """One pass over the VCF -> chrom, pos, ref, alt, {sample: ALT fraction}. Biallelic single-base A/C/G/T SNPs only;
    one contig; all records sharing a position are dropped."""
    samples = list(dict.fromkeys(samples))
    pos_a, ref_l, alt_l = array.array("q"), [], []
    cols = {s: array.array("f") for s in samples}
    col, chrom = {}, None
    with (gzip.open(vcf, "rt") if str(vcf).endswith(".gz") else open(vcf)) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                f = line.rstrip("\n").split("\t")
                miss = [s for s in samples if s not in f[9:]]
                if miss:
                    sys.exit(f"samples not in the VCF header: {', '.join(miss[:5])}")
                col = {s: f.index(s) for s in samples}
                continue
            if not col:
                sys.exit(f"{vcf}: no #CHROM header line before the records")
            f = line.rstrip("\n").split("\t")
            if chrom is None:
                chrom = f[0]
            elif f[0] != chrom:
                sys.exit(f"{vcf} holds more than one contig ({chrom}, {f[0]}, ...); run one contig at a time")
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            pos_a.append(int(f[1])); ref_l.append(ref); alt_l.append(alt)
            for s in samples:
                cols[s].append(_alt_fraction(f[col[s]]))
    pos = np.frombuffer(pos_a, np.int64).copy()
    ref, alt = np.array(ref_l, "<U1"), np.array(alt_l, "<U1")
    freq = {s: np.frombuffer(cols[s], np.float32).copy() for s in samples}
    o = np.argsort(pos, kind="mergesort")
    keep = np.ones(len(pos), bool)
    if len(pos) > 1:
        same = np.diff(pos[o]) == 0
        keep[1:] &= ~same; keep[:-1] &= ~same
    o = o[keep]
    return chrom, pos[o], ref[o], alt[o], {s: v[o] for s, v in freq.items()}, len(pos)


_IUPAC = {frozenset("A"): "A", frozenset("C"): "C", frozenset("G"): "G", frozenset("T"): "T",
          frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("CG"): "S", frozenset("AT"): "W",
          frozenset("GT"): "K", frozenset("AC"): "M"}


def iupac_seq(f, ref, alt):
    """IUPAC sequence of one sample (ALT fractions f): hom REF/ALT -> the base, het -> ambiguity code, uncalled -> N."""
    ch = np.empty(len(f), "<U1")
    hr, ha = f == 0.0, f == 1.0
    het = np.isfinite(f) & ~hr & ~ha
    ch[hr] = ref[hr]; ch[ha] = alt[ha]
    if het.any():
        ch[het] = [_IUPAC.get(frozenset(r + a), "N") for r, a in zip(ref[het], alt[het])]
    ch[~np.isfinite(f)] = "N"
    return "".join(ch.tolist())


# ---- pairs and windows
def select_pairs(p1, p3, k, order, seed):
    pairs = [(x, y) for x in p1 for y in p3]
    if order == "nested":
        rest = pairs[1:]; random.Random(seed).shuffle(rest)
        return ([pairs[0]] + rest)[:k]
    return random.Random(seed).sample(pairs, k) if k < len(pairs) else pairs


def recipient_jobs(sites, keep, r, p1, p3, og, pairs, w):
    """One recipient's alignments in production job order (window, then pair): yields (row, job), row = (start, end,
    sister, donor), job = (job id, 4 sequences, 4 taxon names). Also returns the usable-site and window counts."""
    pos, ref, alt, freq = sites
    usable = keep.copy()
    for s in (p1[0], r, p3[0], og):
        usable &= np.isfinite(freq[s])
    idx = np.flatnonzero(usable)
    starts = range(0, len(idx) - w + 1, w)
    seq = {s: iupac_seq(freq[s][idx], ref[idx], alt[idx])
           for s in dict.fromkeys([r, og] + [x for pr in pairs for x in pr])} if len(starts) else {}
    wp = pos[idx]

    def gen():
        for wi, s0 in enumerate(starts):
            s1 = s0 + w
            st, en = int(wp[s0]), int(wp[s1 - 1])
            for a, b in pairs:
                yield (st, en, a, b), (f"{r}__{wi}__{a}__{b}",
                                       [seq[a][s0:s1], seq[r][s0:s1], seq[b][s0:s1], seq[og][s0:s1]], [a, r, b, og])
    return gen(), len(idx), len(starts)


# ---- IQ-TREE
def topology_4tax(nwk, p1, p2, p3, og):
    """Unrooted 4-taxon Newick -> T1 (p1,p2 sister) / T2 (p2,p3) / T3 (p1,p3) given the outgroup, else None."""
    s = re.sub(r":[0-9.eE+\-]+", "", nwk)
    s = re.sub(r"\)[0-9.eE+\-]*", ")", s)
    for topo, prs in (("T1", [(p1, p2), (p2, p1), (p3, og), (og, p3)]),
                      ("T2", [(p2, p3), (p3, p2), (p1, og), (og, p1)]),
                      ("T3", [(p1, p3), (p3, p1), (p2, og), (og, p2)])):
        for a, b in prs:
            if re.search(r"\(" + re.escape(a) + r"," + re.escape(b) + r"\)", s):
                return topo
    return None


def run_batch(task):
    """One IQ-TREE -S call over a batch of 4-taxon alignments (one file each; -S infers a separate tree per file).
    -> (calls aligned with the batch, ok); a failed call leaves the whole batch None (NA), as in production."""
    bi, jobs, iqtree_bin, seed = task
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", str(bi))
    din, dout = tempfile.mkdtemp(prefix=f"iqbatch_{tag}_in_"), tempfile.mkdtemp(prefix=f"iqbatch_{tag}_out_")
    res = {jid: None for jid, _, _ in jobs}
    ok = False
    try:
        taxa_of = {}
        for jid, seqs, taxa in jobs:
            taxa_of[f"{jid}.fasta"] = taxa
            with open(os.path.join(din, f"{jid}.fasta"), "w") as fh:
                for name, sq in zip(taxa, seqs):
                    fh.write(f">{name}\n{sq}\n")
        prefix = os.path.join(dout, "batch")
        subprocess.run([iqtree_bin, "-S", din, "-st", "DNA", "-m", "GTR", "--fast", "-T", "1", "--quiet", "--redo",
                        "-seed", str(seed), "--prefix", prefix], capture_output=True, timeout=max(120, 10 * len(jobs)))
        if os.path.exists(prefix + ".iqtree") and os.path.exists(prefix + ".treefile"):
            lines = open(prefix + ".iqtree").read().splitlines()
            i0 = next((i for i, ln in enumerate(lines) if ln.strip().startswith("ID")), None)
            if i0 is not None:
                name_of = {}                          # partition ID -> alignment file (the report's ID table)
                for ln in lines[i0 + 1:]:
                    f = ln.split()
                    if not f or not f[0].isdigit():
                        break
                    name_of[int(f[0])] = f[-1]
                trees = open(prefix + ".treefile").read().strip().splitlines()
                for pid, nwk in enumerate(trees, start=1):
                    fn = name_of.get(pid)
                    if fn is None or fn not in taxa_of:
                        continue
                    jid = fn[:-len(".fasta")]
                    if jid in res:
                        res[jid] = topology_4tax(nwk, *taxa_of[fn])
                ok = True
    except Exception as e:                            # noqa: BLE001 -- production behaviour: the batch stays NA
        print(f"[iqtree] batch {bi}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(din, ignore_errors=True); shutil.rmtree(dout, ignore_errors=True)
    return [res[jid] for jid, _, _ in jobs], ok


def stream_batches(ex, batches, ahead):
    """(rows, (calls, ok)) per batch in batch order, at most `ahead` batches in flight."""
    q = deque()
    for rows, task in itertools.islice(batches, ahead):
        q.append((rows, ex.submit(run_batch, task)))
    while q:
        rows, fut = q.popleft()
        nxt = next(batches, None)
        if nxt is not None:
            q.append((nxt[0], ex.submit(run_batch, nxt[1])))
        yield rows, fut.result()


def main():
    a = parse_args()
    if a.combos < 1 or a.snp_window < 1 or a.batch_size < 1:
        sys.exit("--combos, --snp-window and --batch-size must be >= 1")
    p1, p3 = read_samples(a.p1), read_samples(a.p3)
    p2 = list(dict.fromkeys(read_samples(a.p2)))
    ogs = read_samples(a.outgroup)
    og = ogs[0]
    if not os.path.exists(a.iqtree_bin):
        sys.exit(f"IQ-TREE binary not found: {a.iqtree_bin}")
    t0 = time.time()
    ver = subprocess.run([a.iqtree_bin, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    print(f"[iqtree] {ver[0] if ver else a.iqtree_bin}", flush=True)
    if len(ogs) > 1:
        print(f"[iqtree] outgroup: {og} (the first of {len(ogs)} listed; the others are not used)", flush=True)
    chrom, pos, ref, alt, freq, n_read = load_sites(a.vcf, p1 + p2 + p3 + [og])
    keep = np.isfinite(freq[og]) & (freq[og] == 0.0)
    pairs = select_pairs(p1, p3, a.combos, a.pair_order, a.seed)
    print(f"[iqtree] {chrom}: {len(pos):,} biallelic SNPs ({n_read - len(pos):,} records at shared positions dropped), "
          f"{int(keep.sum()):,} with {og} hom-REF; {len(p2)} recipient(s) x {len(pairs)} of {len(p1) * len(p3)} "
          f"(sister, donor) pairs ({a.pair_order}, seed {a.seed}); {a.snp_window}-SNP windows; {a.batch_size:,} "
          f"alignments per IQ-TREE call, {a.n_workers} workers", flush=True)
    print(f"[iqtree] pair order: {', '.join(f'{x}x{y}' for x, y in pairs[:10])}{' ...' if len(pairs) > 10 else ''}",
          flush=True)
    sites = (pos, ref, alt, freq)

    def all_jobs():
        for r in p2:
            g, n_use, n_win = recipient_jobs(sites, keep, r, p1, p3, og, pairs, a.snp_window)
            print(f"[iqtree] {r}: {n_use:,} usable sites -> {n_win:,} windows x {len(pairs)} pairs", flush=True)
            for row, job in g:
                yield (r,) + row, job

    def batches():
        it = all_jobs()
        for bi in itertools.count():
            chunk = list(itertools.islice(it, a.batch_size))
            if not chunk:
                return
            yield [c[0] for c in chunk], (bi, [c[1] for c in chunk], a.iqtree_bin, a.iqtree_seed)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    n_rows = n_na = n_bad = nb = 0
    cnt = dict(T1=0, T2=0, T3=0)
    with gzip.open(tmp, "wt") as fh, ProcessPoolExecutor(a.n_workers) as ex:
        fh.write("chrom\tstart\tend\trecipient\tsister\tdonor\ttopology\n")
        for rows, (calls, ok) in stream_batches(ex, batches(), 2 * a.n_workers):
            nb += 1
            if not ok:
                n_bad += 1
                print(f"[iqtree] WARNING batch {nb} ({len(rows):,} alignments) failed: all NA", flush=True)
            for (r, st, en, x, y), c in zip(rows, calls):
                fh.write(f"{chrom}\t{st}\t{en}\t{r}\t{x}\t{y}\t{c or 'NA'}\n")
                if c is None:
                    n_na += 1
                else:
                    cnt[c] += 1
            n_rows += len(rows)
            if nb % max(1, a.n_workers) == 0:
                print(f"[iqtree]   {nb} batches, {n_rows:,} alignments ({time.time() - t0:.0f}s)", flush=True)
    os.replace(tmp, a.out)
    print(f"[iqtree] done in {time.time() - t0:.0f}s: {n_rows:,} rows ({nb} IQ-TREE calls, {n_bad} failed) -> {a.out}; "
          f"T1 {cnt['T1']:,} T2 {cnt['T2']:,} T3 {cnt['T3']:,} NA {n_na:,}", flush=True)


if __name__ == "__main__":
    main()
