"""Input handling: sample lists, the VCF (one contig, biallelic SNPs), the exclusion mask, unsupported spans, the
genetic map. Coordinates are 1-based VCF positions throughout; the mask BED is 0-based half-open on input."""
import gzip
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def read_samples(path):
    """Sample names, whitespace-separated, from a file."""
    return open(path).read().split()


class SiteData:
    """Biallelic SNPs of one contig: pos (int64) and, per sample, the ALT allele fraction of its genotype
    (0, 0.5, 1 for diploids; NaN when any allele is uncalled)."""
    __slots__ = ("chrom", "pos", "freq")

    def __init__(self, chrom, pos, freq):
        self.chrom, self.pos, self.freq = chrom, pos, freq

    def __len__(self):
        return len(self.pos)


def _alt_fraction(gt):
    """GT string (before the first ':') -> fraction of non-REF alleles, NaN if any allele is missing."""
    alleles = gt.split("|" if "|" in gt else "/")
    d = 0
    for a in alleles:
        if a == ".":
            return np.nan
        if a != "0":
            d += 1
    return d / len(alleles)


def _finish(path, chrom, pos, freq, n_total, n_notsnp, t0, engine, log):
    """sort by position, drop every record of a position held by several records -> SiteData"""
    order = np.argsort(pos, kind="mergesort")
    if not np.array_equal(order, np.arange(len(order))):
        pos = pos[order]
        freq = {s: f[order] for s, f in freq.items()}
    n_dup = 0
    if len(pos) > 1:
        same = np.diff(pos) == 0
        if same.any():
            dup = np.zeros(len(pos), bool); dup[1:] |= same; dup[:-1] |= same
            n_dup = int(dup.sum())
            pos = pos[~dup]
            freq = {s: f[~dup] for s, f in freq.items()}
    log(f"VCF {path}: contig {chrom}, {n_total:,} records, {n_notsnp:,} not biallelic SNPs"
        + (f", {n_dup:,} dropped (duplicate position)" if n_dup else "") + f" -> {len(pos):,} SNPs "
        f"({time.time() - t0:.1f}s, {engine})")
    if not len(pos):
        raise SystemExit(f"{path}: no biallelic SNPs" + (f" on {chrom}" if chrom else ""))
    return SiteData(chrom, pos, freq)


def _load_vcf_python(path, samples, contig, log):
    opener = gzip.open if str(path).endswith(".gz") else open
    t0 = time.time()
    pos_l, cols, col_idx, cache = [], {s: [] for s in samples}, {}, {}
    chrom = contig
    n_total = n_notsnp = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                miss = [s for s in samples if s not in fields[9:]]
                if miss:
                    raise SystemExit(f"samples not in the VCF header: {', '.join(miss[:5])}")
                col_idx = {s: fields.index(s) for s in samples}
                continue
            if not col_idx:
                raise SystemExit(f"{path}: no #CHROM header line")
            fields = line.rstrip("\n").split("\t")
            if contig is not None:
                if fields[0] != contig:
                    continue
            elif chrom is None:
                chrom = fields[0]
            elif fields[0] != chrom:
                raise SystemExit(f"{path} holds more than one contig ({chrom}, {fields[0]}, ...); give the contig to read")
            n_total += 1
            ref, alt = fields[3], fields[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                n_notsnp += 1
                continue
            pos_l.append(int(fields[1]))
            for s in samples:
                gt = fields[col_idx[s]].split(":", 1)[0]
                v = cache.get(gt)
                if v is None:
                    v = cache[gt] = _alt_fraction(gt)
                cols[s].append(v)
    return _finish(path, chrom, np.asarray(pos_l, np.int64), {s: np.asarray(cols[s], np.float32) for s in samples},
                   n_total, n_notsnp, t0, "python", log)


_ACGT = np.zeros(256, bool); _ACGT[[ord(c) for c in "ACGT"]] = True


def _chunk_values(buf, S):
    """One block of bcftools query lines (CHROM POS REF ALT GT x S, tab-separated) -> (CHROM token start/length of
    every line, POS, biallelic-SNP mask, ALT fractions (lines x S, float32)). GT tokens of the usual shapes are decoded
    with array operations (a/b, a|b: two alleles; a: haploid; '.' allele -> NaN; an allele other than '0' counts as
    non-REF); any other token goes through _alt_fraction, so the rules are those of the Python reader."""
    b = np.frombuffer(buf, np.uint8)
    ends = np.flatnonzero((b == 9) | (b == 10))
    starts = np.r_[0, ends[:-1] + 1]
    n = int((b == 10).sum()); W = 4 + S
    if len(ends) != n * W:
        raise RuntimeError("unexpected bcftools query output (fields per line)")
    T = starts.reshape(n, W); L = (ends - starts).reshape(n, W)
    pl = L[:, 1]; pos = np.zeros(n, np.int64)
    for k in range(int(pl.max()) if n else 0):
        has = pl > k
        pos = np.where(has, pos * 10 + (b[np.minimum(T[:, 1] + k, len(b) - 1)].astype(np.int64) - 48), pos)
    snp = (L[:, 2] == 1) & _ACGT[b[T[:, 2]]] & (L[:, 3] == 1) & _ACGT[b[T[:, 3]]]
    G, GL = T[:, 4:], L[:, 4:]
    top = len(b) - 1
    c0 = b[G]; c1 = b[np.minimum(G + 1, top)]; c2 = b[np.minimum(G + 2, top)]
    v = np.full(G.shape, np.nan, np.float32)
    one = GL == 1
    v[one & (c0 != 46)] = (c0[one & (c0 != 46)] != 48).astype(np.float32)
    two = (GL == 3) & ((c1 == 47) | (c1 == 124))
    ok2 = two & (c0 != 46) & (c2 != 46)
    v[ok2] = ((c0[ok2] != 48).astype(np.float32) + (c2[ok2] != 48).astype(np.float32)) / 2
    odd = ~(one | two)
    if odd.any():
        for i, j in zip(*np.nonzero(odd)):
            v[i, j] = _alt_fraction(bytes(b[G[i, j]:G[i, j] + GL[i, j]]).decode())
    return T[:, 0], L[:, 0], pos, snp, v, b


def _load_vcf_bcftools(bcf, path, samples, contig, log):
    """bcftools streams CHROM, POS, REF, ALT and the GT of the wanted samples only; _chunk_values decodes them with
    array operations, with the rules of the Python reader (same result, checked bit for bit)."""
    t0 = time.time()
    hdr = subprocess.run([bcf, "query", "-l", path], capture_output=True, text=True)
    if hdr.returncode:
        raise RuntimeError(hdr.stderr.strip()[:300])
    in_vcf = hdr.stdout.split()
    miss = [s for s in samples if s not in in_vcf]
    if miss:
        raise SystemExit(f"samples not in the VCF header: {', '.join(miss[:5])}")
    cmd = [bcf, "query", "-H", "-s", ",".join(s for s in in_vcf if s in set(samples)),
           "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n"]
    if contig is not None:
        indexed = any(os.path.exists(path + x) for x in (".tbi", ".csi"))
        cmd += ["-r" if indexed else "-t", contig]
    proc = subprocess.Popen(cmd + [path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    head = proc.stdout.readline().decode().rstrip("\n").split("\t")
    order = [h.split("]", 1)[1].rsplit(":", 1)[0] for h in head[4:]]          # "[5]NAME:GT" -> NAME
    S = len(order)
    chrom, pos_l, cols, n_total, rest = None, [], [], 0, b""
    while True:
        blk = proc.stdout.read(64 << 20)
        data = rest + blk
        cut = data.rfind(b"\n") + 1
        if cut == 0 and blk:
            rest = data; continue
        buf, rest = data[:cut], data[cut:]
        if buf:
            cs, cl, pos, snp, v, bb = _chunk_values(buf, S)
            if chrom is None:
                chrom = bytes(bb[cs[0]:cs[0] + cl[0]]).decode()
            if contig is None:
                first = np.frombuffer(chrom.encode(), np.uint8)
                same = cl == len(first)
                for k in range(len(first)):
                    same &= bb[np.minimum(cs + k, len(bb) - 1)] == first[k]
                if not same.all():
                    proc.kill()
                    i = int(np.argmin(same))
                    raise SystemExit(f"{path} holds more than one contig ({chrom}, "
                                     f"{bytes(bb[cs[i]:cs[i] + cl[i]]).decode()}, ...); give the contig to read")
            n_total += len(pos); pos_l.append(pos[snp]); cols.append(v[snp])
        if not blk:
            break
    err = proc.stderr.read().decode(); rc = proc.wait()
    if rc:
        raise RuntimeError(err.strip()[:300])
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    V = np.concatenate(cols) if cols else np.empty((0, S), np.float32)
    freq = {s: np.ascontiguousarray(V[:, order.index(s)]) for s in samples}
    chrom = contig if contig is not None else chrom
    return _finish(path, chrom, pos, freq, n_total, n_total - len(pos), t0, "bcftools", log)


def load_vcf(path, samples, contig=None, log=print):
    """Biallelic SNPs (single-base A/C/G/T REF and ALT) of one contig: the VCF's only contig, or `contig` of a
    multi-contig VCF (read through its index when there is one). Records sharing a position are all dropped. Uses
    bcftools when it is on PATH (only the wanted samples are parsed), else a pure-Python reader; same result.
    -> SiteData."""
    samples = list(dict.fromkeys(samples))
    here = os.path.join(os.path.dirname(sys.executable), "bcftools")     # the environment's own bcftools first
    bcf = here if os.access(here, os.X_OK) else shutil.which("bcftools")
    if bcf:
        try:
            return _load_vcf_bcftools(bcf, path, samples, contig, log)
        except RuntimeError as e:
            log(f"[vcf] bcftools could not read {path} ({e}); reading it in Python")
    return _load_vcf_python(path, samples, contig, log)


def load_mask(path, chrom, log=print):
    """BED (0-based half-open, optionally gzipped) of regions to EXCLUDE -> merged (starts, ends) int64 arrays, 0-based
    half-open. Only the rows of `chrom` are used when the BED names it (all rows otherwise, as the old core did)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3:
                rows.append((f[0], int(f[1]), int(f[2])))
    names = {r[0] for r in rows}
    if chrom in names:
        iv = sorted((s, e) for c, s, e in rows if c == chrom)
    else:
        iv = sorted((s, e) for _, s, e in rows)
        if rows:
            log(f"[mask] {path} does not name contig {chrom}: all {len(iv):,} interval(s) used")
    if not iv:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array([m[0] for m in merged], np.int64), np.array([m[1] for m in merged], np.int64)


def apply_mask(sd, mask, log=print):
    """Drop the SNPs inside mask intervals (0-based half-open)."""
    p0 = sd.pos - 1
    i = np.searchsorted(mask[1], p0, side="right")
    inside = (i < len(mask[0])) & (mask[0][np.minimum(i, len(mask[0]) - 1)] <= p0)
    if inside.any():
        sd.pos = sd.pos[~inside]
        sd.freq = {s: f[~inside] for s, f in sd.freq.items()}
    log(f"[mask] {len(mask[0]):,} interval(s); {int(inside.sum()):,} SNPs inside dropped (masked bp are not counted in "
        f"gene-tree windows and are NA in the tracts)")
    return sd


def unsupported_spans(pos, mask, gap_bp):
    """Merged (starts, ends), 1-based inclusive, of bp without support: the mask plus SNP-free stretches longer than
    gap_bp (half of gap_bp is left supported on each side; a leading gap counts from position 1). None if empty."""
    iv_s, iv_e = [], []
    if mask is not None and len(mask[0]):
        iv_s.append(np.asarray(mask[0], np.int64) + 1); iv_e.append(np.asarray(mask[1], np.int64))
    pos = np.asarray(pos, np.int64)
    if gap_bp > 0 and len(pos):
        h = gap_bp // 2
        a = np.r_[0, pos[:-1]]; b = pos
        big = b - a - 1 > np.where(a == 0, h, gap_bp)
        if big.any():
            s = np.where(a[big] == 0, 1, a[big] + 1 + h); e = b[big] - 1 - h
            ok = e >= s
            iv_s.append(s[ok]); iv_e.append(e[ok])
    if not iv_s or not sum(len(x) for x in iv_s):
        return None
    s = np.concatenate(iv_s); e = np.concatenate(iv_e)
    o = np.argsort(s, kind="mergesort"); s, e = s[o], e[o]
    ms, me = [int(s[0])], [int(e[0])]
    for x, y in zip(s[1:], e[1:]):
        if x <= me[-1] + 1:
            me[-1] = max(me[-1], int(y))
        else:
            ms.append(int(x)); me.append(int(y))
    return np.array(ms, np.int64), np.array(me, np.int64)


def unsupported_overlap(starts, ends, unsup):
    """Unsupported bp inside each [start, end] (1-based inclusive)."""
    us, ue = unsup
    cum = np.r_[0, np.cumsum(ue - us + 1)]

    def upto(x):
        x = np.asarray(x, np.int64)
        k = np.searchsorted(us, x, side="right"); j = np.maximum(k - 1, 0)
        return np.where(k > 0, cum[j] + np.minimum(x, ue[j]) - us[j] + 1, 0)
    return (upto(ends) - upto(np.asarray(starts, np.int64) - 1)).astype(float)


class GeneticMap:
    """PLINK map (chrom id cM bp) -> effective bp: genetic position / mean rate, so 1 effective bp = the contig's mean
    cM per bp. Positions between map points are interpolated linearly; cM is made non-decreasing."""

    def __init__(self, path, log=print):
        m = pd.read_csv(path, sep=r"\s+", header=None, usecols=[2, 3], names=["cM", "bp"])
        m = m.drop_duplicates("bp").sort_values("bp")
        self.bp, self.cm = m.bp.values.astype(float), np.maximum.accumulate(m.cM.values.astype(float))
        self.rate = (self.cm[-1] - self.cm[0]) / max(self.bp[-1] - self.bp[0], 1.0)
        self.path = os.path.abspath(path)
        log(f"[map] {len(self.bp):,} map points, {self.cm[-1] - self.cm[0]:.4f} cM over {self.bp[-1] - self.bp[0]:,.0f} bp")

    def eff(self, x):
        return np.interp(np.asarray(x, float), self.bp, self.cm) / self.rate
