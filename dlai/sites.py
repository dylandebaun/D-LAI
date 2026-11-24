"""Sites of one recipient: polarisation, site patterns, the informative-site filter and the emission code.

Letters per sample (derived-allele fraction, ancestral = REF): A = 0, B = 1, H = anything else (heterozygous or
uncalled). A site pattern is (P1 letter, P2 letter, P3 letter, outgroup A), e.g. ABBA. The informative-site filter keeps
a site when some (P1 sample, P3 sample) pair gives a pattern of the alphabet (gate_alphabet; with the production gate
threshold every ABBA-like member survives), plus the sites with the P1 panel uncalled that some P1 genotype would admit
(the "P1 missing" emission bin). The emission code of a site is the recipient letter x P1-panel derived-frequency bin
(0, <= 1/2, < 1, 1) x P3-panel bin (0, < 1, 1): 36 codes, + 24 codes with the P1 panel uncalled (60 in total)."""
from collections import namedtuple

import numpy as np
import pandas as pd

LETTERS = ("A", "B", "H")
_NAMED = ["ABBA", "AHBA", "BHAA", "ABHA", "BHBA", "AHHA", "BABA", "BHHA", "AABA", "BBAA", "AHAA", "BAHA"]
_PAT_STR = {("A", "B", "B"): "ABBA", ("A", "H", "B"): "AHBA", ("B", "H", "A"): "BHAA", ("A", "B", "H"): "ABHA",
            ("B", "H", "B"): "BHBA", ("A", "H", "H"): "AHHA", ("B", "A", "B"): "BABA", ("B", "H", "H"): "BHHA",
            ("A", "A", "B"): "AABA", ("B", "B", "A"): "BBAA", ("A", "H", "A"): "AHAA", ("B", "A", "H"): "BAHA"}
for _a in LETTERS:
    for _b in LETTERS:
        for _c in LETTERS:
            _PAT_STR.setdefault((_a, _b, _c), _a + _b + _c + "A")
PATTERN_CODES = _NAMED + sorted(set(_PAT_STR.values()) - set(_NAMED)) + ["other"]
N_CODES = len(PATTERN_CODES)
OTHER_CODE = N_CODES - 1
PATTERN_MAP = {p: i for i, p in enumerate(PATTERN_CODES[:-1])}
PATTERN_LUT = np.full((3, 3, 3), OTHER_CODE, np.int16)            # [P1 letter, P2 letter, P3 letter] -> pattern code
for (_a, _b, _c), _p in _PAT_STR.items():
    PATTERN_LUT[LETTERS.index(_a), LETTERS.index(_b), LETTERS.index(_c)] = PATTERN_MAP.get(_p, OTHER_CODE)

ABBA_LIKE = ["ABBA", "AHBA", "ABHA", "HBBA", "AHHA", "HBHA"]
CANDIDATE_ALPHABET = "ABBA+AHBA+ABHA+HBBA+AHHA+HBHA AABA ABAA BABA BBAA HABA HBAA"
EXCLUDE_DELTA = -1000.0          # ABBA-like member gate threshold (production)

Alphabet = namedtuple("Alphabet", ["string", "labels", "other_code", "lut"])   # lut: pattern code -> alphabet code


def letter_idx(f):
    """derived-allele fraction -> letter index 0 = A (0), 1 = B (1), 2 = H (else, NaN included)."""
    idx = np.full(f.shape, 2, np.int8)
    idx[f == 0.0] = 0
    idx[f == 1.0] = 1
    return idx


def polarise(sd, outgroup, log=print):
    """Ancestral = REF: keep the sites where the outgroup is called homozygous REF."""
    f = sd.freq[outgroup]
    keep = np.isfinite(f) & (f == 0.0)
    log(f"[polarise] outgroup {outgroup} homozygous REF at {int(keep.sum()):,} of {len(f):,} sites")
    return keep


def _parse_alphabet(s):
    groups = []
    for token in s.split():
        members = token.split("+")
        for p in members:
            if len(p) != 4 or any(c not in "ABH" for c in p[:3]) or p[3] != "A":
                raise ValueError(f"invalid pattern {p!r} in alphabet {s!r}")
        groups.append((token, frozenset(members)))
    return groups


def make_alphabet(s):
    groups = _parse_alphabet(s)
    other = len(groups)
    to_new = {}
    for i, (_, ps) in enumerate(groups):
        for p in ps:
            to_new[p] = i
    lut = np.array([to_new.get(p, other) for p in PATTERN_CODES[:-1]] + [other], np.int64)
    return Alphabet(s, [g for g, _ in groups] + ["other"], other, lut)


def pooled_pair_counts(c1, l2, c3, n1, n3):
    """Number of (site, P1 sample, P3 sample) cells with each pattern code (uncalled members -> other)."""
    pooled = np.zeros(N_CODES, np.int64)
    for a in range(3):
        for b in range(3):
            w = c1[:, a] * c3[:, b]
            pooled += np.bincount(PATTERN_LUT[a, l2, b], weights=w, minlength=N_CODES).round().astype(np.int64)
    pooled[OTHER_CODE] += len(l2) * n1 * n3 - pooled.sum()
    return pooled


def gate_alphabet(pooled, n_cells, pi_int, log=print):
    """ABBA-like members whose INT signal (pattern frequency minus that of its P1/P2-swapped partner, over pi_INT) is
    above EXCLUDE_DELTA are lumped into one group; the other candidate patterns are kept as their own groups."""
    survivors = []
    log(f"[gate] ABBA-like member gate (delta={EXCLUDE_DELTA:g}, pi_I={pi_int:.4f}):")
    for m in ABBA_LIKE:
        c = m[1] + m[0] + m[2] + m[3]
        pm, pc = pooled[PATTERN_MAP[m]] / max(n_cells, 1), pooled[PATTERN_MAP[c]] / max(n_cells, 1)
        x = (pm - pc) / max(pi_int, 1e-9)
        log(f"    {m:<5} vs {c:<5}  INT={x:+.5f}  (pbar_m={pm:.5f} pbar_c={pc:.5f})  -> {'keep' if x > EXCLUDE_DELTA else 'DROP'}")
        if x > EXCLUDE_DELTA:
            survivors.append(m)
    rest = [g for g in CANDIDATE_ALPHABET.split()[1:]]
    s = " ".join((["+".join(survivors)] if survivors else []) + rest)
    log(f"[gate] alphabet: {s}")
    return make_alphabet(s)


def panel_stats(sd, p1, p3):
    """Per-site panel summaries shared by every recipient: letter counts (A, B, H) of the P1 and P3 panels, their mean
    derived fractions, and the letters of the reference pair p1[0] x p3[0]."""
    def letter_counts(panel):
        c = np.zeros((len(sd.pos), 3), np.int64)
        for s in panel:
            d = sd.freq[s]
            li = letter_idx(d); ok = np.isfinite(d)
            for k in range(3):
                c[:, k] += ok & (li == k)
        return c
    with np.errstate(invalid="ignore"):
        fq1 = np.nanmean(np.stack([sd.freq[s] for s in p1]), 0)
        fq3 = np.nanmean(np.stack([sd.freq[s] for s in p3]), 0)
    d1, d3 = sd.freq[p1[0]], sd.freq[p3[0]]
    return dict(c1=letter_counts(p1), c3=letter_counts(p3), fq1=fq1, fq3=fq3, l1=letter_idx(d1), l3=letter_idx(d3),
                ok13=np.isfinite(d1) & np.isfinite(d3))


def recipient_sites(sd, keep, p1, r, p3, gt_pi_int, panels, log=print):
    """-> (DataFrame of the recipient's decoded sites, alphabet). Columns: pos, ref_code (alphabet code of the reference
    pair p1[0] x p3[0]), fq1 / fq3 (P1 / P3 panel mean derived fraction, NaN if the panel is uncalled), l2 (recipient
    letter), dist (bp to the previous site; 1 for the first). panels: panel_stats(sd, p1, p3)."""
    d2 = sd.freq[r]
    use = keep & np.isfinite(d2)
    n_drop = int((keep & ~use).sum())
    l2 = letter_idx(d2[use])
    c1, c3 = panels["c1"][use], panels["c3"][use]
    ref = PATTERN_LUT[panels["l1"][use], l2, panels["l3"][use]].astype(np.int16)
    ref[~panels["ok13"][use]] = OTHER_CODE
    fq1, fq3 = panels["fq1"][use], panels["fq3"][use]
    pos = sd.pos[use]
    log(f"  sites polarised and called in {r}: {len(pos):,} (dropped {n_drop:,} where {r} is uncalled)")
    if len(pos) < 100:
        raise ValueError(f"only {len(pos)} usable sites for recipient {r}")
    alph = gate_alphabet(pooled_pair_counts(c1, l2, c3, len(p1), len(p3)), len(l2) * len(p1) * len(p3),
                         max(gt_pi_int, 1e-4), log)
    # informative-site filter: some (P1 sample, P3 sample) pair gives an alphabet pattern
    rl = alph.lut
    inf = np.zeros(len(l2), bool)
    for a in range(3):
        for b in range(3):
            inf |= (c1[:, a] > 0) & (c3[:, b] > 0) & (rl[PATTERN_LUT[a, l2, b]] != alph.other_code)
    # + sites with a panel uncalled: P3 called, the recipient or the donor panel derived, and some P1 genotype would
    # make the site an alphabet pattern
    miss = ~np.isfinite(fq1) | ~np.isfinite(fq3)
    extra = ~inf & miss & ((l2 > 0) | (np.nan_to_num(fq3, nan=0.0) > 0)) & np.isfinite(fq3)
    l2x, l3x = np.clip(l2, 0, 2), letter_idx(np.nan_to_num(fq3, nan=0.0))
    adm = np.zeros(len(l2), bool)
    for g1 in range(3):
        adm |= rl[PATTERN_LUT[g1, l2x, l3x]] != alph.other_code
    log(f"  [miss-bin] {int((extra & ~adm).sum()):,} P1-uncalled sites outside any P1 completion dropped; keeping "
        f"{int((extra & adm).sum()):,} extra sites with a panel uncalled")
    k = inf | (extra & adm)
    df = pd.DataFrame(dict(pos=pos[k], ref_code=rl[ref[k]], fq1=fq1[k], fq3=fq3[k], l2=l2[k]))
    df["dist"] = df["pos"].diff().fillna(1).astype(int)
    log(f"  alphabet filter: {len(df):,} sites kept / {len(pos):,} ({100 * len(df) / len(pos):.1f}%)")
    return df, alph


def emission_codes(df):
    """-> (code per site, number of codes): recipient letter x P1 bin x P3 bin (36), plus 'P1 missing' codes 36-59
    when some site has a panel uncalled."""
    f1, f3, l2 = df["fq1"].values, df["fq3"].values, df["l2"].values.astype(int)
    b1 = np.select([f1 == 0, f1 <= 0.5, f1 < 1], [0, 1, 2], 3); b3 = np.select([f3 == 0, f3 < 1], [0, 1], 2)
    code = l2 * 12 + b1 * 3 + b3
    okc = np.isfinite(f1) & np.isfinite(f3)
    if okc.all():
        return code, 36
    b1 = np.where(np.isfinite(f1), b1, 4); b3 = np.where(np.isfinite(f3), b3, 3)
    code = np.where(okc, code, 36 + l2 * 8 + np.where(b1 == 4, b3, 4 + np.minimum(b1, 3)))
    return code, 60


def code_label(c):
    """recipient letter + P1 bin + P3 bin; 'M' = panel uncalled (codes 36-59)."""
    if c < 36:
        return f"{'ABH'[c // 12]}{(c % 12) // 3}{c % 3}"
    l, o = divmod(c - 36, 8)
    return "ABH"[l] + ("MM" if o == 3 else f"M{o}" if o < 3 else f"{o - 4}M")


def donor_private(df):
    """recipient derived, no P1-panel member derived, the P3 panel derived (both panels called)."""
    f1, f3 = df["fq1"].values, df["fq3"].values
    return (df["l2"].values > 0) & (f1 == 0) & (f3 > 0) & np.isfinite(f1) & np.isfinite(f3)
