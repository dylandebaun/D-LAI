"""dlai decode: one contig, several recipients.

  1. read the VCF once (biallelic SNPs; mask applied), polarise by the outgroup, read the gene trees
  2. per recipient: decoded sites, emission codes, window labels (fixed for the whole run)
  3. decode 1 -> <out>/dlai_v2/unweighted/: unweighted, no genetic map; with >= 2 recipients the emission is fitted from
     the counts pooled over every recipient. Its diagnostics are the reported ones.
  4. design-effect weights from decode 1 (weights.py) -> <out>/dlai_v2/weights/
  5. decode 2 -> <out>/dlai_v2/: the weights scale the training counts and the log-emissions; with --genetic-map the
     transitions and the anchor run lengths are measured in map-effective bp (TD), without it in bp (Db)
Per decode and recipient: four-state emission + generator (labels.anchor_q4) -> forward-backward -> posteriors
(Other / INT / ILS = ILS_RD + ILS_SD) -> tracts; copy-number annotation P(two copies | INT) from a lumped copy-number
generator (calls and P(INT) stay the four-state decode)."""
import io
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import emission as em
from . import hmm
from . import labels as lb
from . import sites as st
from . import weights as wt
from .genetrees import read_gene_trees, window_distances
from .io import GeneticMap, apply_mask, load_mask, load_vcf, read_samples, unsupported_spans

POOL_MIN = 2
CN_SWITCH = 1.0                   # copy-number switch rate within a block, x the block's exit rate
_G = {}                           # run state shared with the worker processes (fork)


class Log:
    """Collects a recipient's log lines (written to its run log) and echoes them when live."""

    def __init__(self, live=False):
        self.buf, self.live = io.StringIO(), live

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        self.buf.write(s + "\n")
        if self.live:
            print(s, flush=True)


# ------------------------------------------------------------------------------------------------ preparation
def prepare(a, log=print):
    p1, p2, p3 = read_samples(a.p1), read_samples(a.p2), read_samples(a.p3)
    og = read_samples(a.outgroup)[0]
    p2 = list(dict.fromkeys(p2))
    sd = load_vcf(a.vcf, list(p1) + p2 + list(p3) + [og], getattr(a, "contig", None), log)
    mask = None
    if a.mask:
        mask = load_mask(a.mask, sd.chrom, log)
        sd = apply_mask(sd, mask, log)
    unsup = unsupported_spans(sd.pos, mask, a.max_site_gap)
    if unsup is not None:
        log(f"[unsupported] {len(unsup[0]):,} span(s), {int((unsup[1] - unsup[0] + 1).sum()):,} bp (mask + SNP-free "
            f"stretches > {a.max_site_gap:,} bp): not counted in gene-tree window bp, NA in the tracts")
    keep = st.polarise(sd, og, log)
    trees = read_gene_trees(a.gene_trees, p2, p1, p3, chrom=sd.chrom, seed=1, keep_windows=a.keep_windows, log=log)
    panels = st.panel_stats(sd, p1, p3)
    rec = {}
    for r in p2:
        L = Log()
        t = window_distances(trees[r], sd, keep, r, p1, p3, og)
        ts = lb.topo_stats(t)
        df, alph = st.recipient_sites(sd, keep, p1, r, p3, ts["pi_int"], panels, L)
        code, nc = st.emission_codes(df)
        lab = lb.window_labels(df.pos.values, df.ref_code.values.astype(int), t, lb.alphabet_codes(alph),
                               st.donor_private(df), unsup, L)
        rec[r] = dict(df=df, alph=alph, code=code, nc=nc, lab=lab, table=t, stats=ts, prep_log=L.buf.getvalue())
        log(f"[prepare] {r}: {len(df):,} decoded sites, {len(t):,} gene-tree windows, {nc} emission codes")
    return dict(p1=p1, p2=p2, p3=p3, og=og, sd=sd, unsup=unsup, rec=rec, chrom=sd.chrom)


# -------------------------------------------------------------------------------------------- one recipient
def decode_recipient(r, D, gmap, w, pooled, diag_pre, want_loglr, live=False):
    """Decode one recipient into D. w: per-site evidence weights (None = unweighted); pooled: cohort counts (None =
    own counts). -> dict(summary, tracts, sites, loglr, theta, log)."""
    t0 = time.time()
    R, trial = _G["rec"][r], _G["trial"]
    log = Log(live)
    log("#" * 78); log(f"# RECIPIENT {r}  -> {D}"); log("#" * 78)
    log(R["prep_log"].rstrip())
    df, code, nc, lab = R["df"], R["code"], R["nc"], R["lab"]
    pos = df.pos.values
    if gmap is not None:
        e = gmap.eff(pos)
        distances = np.r_[1, np.maximum(np.rint(np.diff(e)), 1)].astype(int)
        log(f"[map] {len(pos):,} sites: mean gap {df.dist.values.mean():.1f} bp -> {distances.mean():.1f} effective bp")
    else:
        distances = df.dist.values.astype(int)
    raw, t3 = em.training_counts(code, nc, lab["ws"], lab["ws4"], w)
    E = em.fit(nc, raw, t3, pooled=pooled, copy_number=True, label=r, log=log)
    em.counts_table(nc, E["cnt"] - 0.5, E["t3"], E["mu"]).to_csv(os.path.join(D, f"mu_{r}.tsv"), sep="\t", index=False)
    if E["int2"] is not None:
        pd.DataFrame({"code": E["labels"], "mu_INT": E["mu"][1], "mu_INT1": E["int2"]["mu1"],
                      "mu_INT2": E["int2"]["mu2"]}).to_csv(os.path.join(D, f"mu_{r}.int2.tsv"), sep="\t", index=False)
    le3 = np.log(E["mu"])[:, code]
    le4 = np.log(E["mu4"])[:, code]
    if w is not None:
        le3 = le3 * w[None, :]; le4 = le4 * w[None, :]
    theta = lb.anchor_theta(lab["gt"], lab["wstate"], gmap, log)
    Q3 = hmm.q_from_theta(theta); pi3 = hmm.stationary(Q3); pi3 = pi3 / pi3.sum(); pi3 = pi3 / pi3.sum()
    Q4 = lb.anchor_q4(lab["gt"], lab["w4"], gmap, log)
    pi4 = hmm.stationary_clipped(Q4)
    log("[decode] 4-state decode, stationary Bg/INT/ILS_RD/ILS_SD = " + "/".join(f"{p:.4f}" for p in pi4))
    g4, ll4, la, lbeta, le = hmm.forward_backward(le4, distances, Q4, pi4)
    p_int2 = None
    if E["int2"] is not None:
        spl = {b: v for b, v in ((1, E["int2"]), (2, E["int2"]["ils"]), (3, E["int2"]["sdt"])) if v is not None}
        row = lambda mu: (np.log(mu)[code] if w is None else np.log(mu)[code] * w)
        rows = [le4[0]] + [row(spl[b]["mu1"]) if b in spl else le4[b] for b in (1, 2, 3)] + [row(spl[b]["mu2"]) for b in sorted(spl)]
        Qc, two = hmm.cn_generator(Q4, {b: float(spl[b]["w_mix"]) for b in spl}, CN_SWITCH)
        mem = {x: [x] + ([two[x]] if x in two else []) for x in range(4)}
        err = max(abs(Qc[i, mem[b]].sum() - Q4[a_, b]) / max(abs(Q4[a_, b]), 1e-300)
                  for a_ in range(4) for b in range(4) if a_ != b for i in mem[a_])
        if err > 1e-9:
            raise SystemExit(f"copy-number generator does not lump to Q4 (max rel. error {err:.2e})")
        gc = hmm.forward_backward(np.vstack(rows), distances, Qc, hmm.stationary_clipped(Qc))[0]
        p_int2 = g4[1] * (gc[two[1]] / np.maximum(gc[1] + gc[two[1]], 1e-300))
        log("[copy number] w (emission) INT/RD/SD = " + "/".join(f"{spl[b]['w_mix']:.4f}" if b in spl else "-" for b in (1, 2, 3))
            + f"; P(two | INT) INT-weighted mean {p_int2.sum() / max(g4[1].sum(), 1e-300):.3f}")
    gamma = hmm.collapse4(g4)
    if diag_pre:
        ll3 = hmm.forward_backward(le3, distances, Q3, pi3)[1]
        ctx = dict(code=code, ws=lab["ws"].copy(), cnt=E["cnt"].copy(), mu=E["mu"].copy(), w=None if w is None else w.copy())
        for suf in ("profile.tsv", "blocks.tsv", "diag.tsv", "FAILED"):
            if os.path.exists(f"{diag_pre}_{r}_{suf}"):
                os.remove(f"{diag_pre}_{r}_{suf}")
        try:
            dg.refit_models(diag_pre, r, ctx, theta, (lab["gt"], lab["wstate"]), le3, distances, float(ll3), log=log)
        except Exception:
            import traceback
            log(traceback.format_exc()); log(f"[diag] {r}: in-core diagnostics FAILED (decode unaffected)")
            with open(f"{diag_pre}_{r}_FAILED", "w") as fh:
                fh.write(traceback.format_exc())
    y_hat = gamma.argmax(axis=0)
    sites = pd.DataFrame(dict(pos=pos, p_other_fe=gamma[0], p_int_fe=gamma[1], p_ils_fe=gamma[2], y_hat_fe=y_hat))
    if p_int2 is not None:
        sites["p_int2_fe"] = p_int2
    sites["p_ils_rdt"] = g4[2]; sites["p_ils_sdt"] = g4[3]
    sites.to_csv(os.path.join(D, f"{r}_sites.csv"), index=False)
    tr = hmm.tracts(pos, y_hat, gamma[1], la, lbeta, le, Q4)
    if p_int2 is not None:
        c2 = np.r_[0.0, np.cumsum(p_int2)]; c1 = np.r_[0.0, np.cumsum(gamma[1])]
        i0 = tr.site_i0.values.astype(int); i1 = tr.site_i1.values.astype(int) + 1
        tr["two_copy_share"] = (c2[i1] - c2[i0]) / np.maximum(c1[i1] - c1[i0], 1e-300)
    if trial["unsup"] is not None:
        tr = hmm.carve_unsupported(tr, pos, trial["unsup"])
        log(f"[unsupported] carved {int((tr.state == -1).sum()):,} NA no-call span(s) into the tracts")
    tr.to_csv(os.path.join(D, f"{r}_tracts.csv"), index=False)
    if lab["window_bp"] is not None:
        lab["window_bp"].to_csv(os.path.join(D, f"dlai_{r}_window_bp.tsv"), sep="\t", index=False)
    R["table"].to_csv(os.path.join(D, f"dlai_{r}_gt_topo_{trial['p1'][0]}_{trial['p3'][0]}_w25.tsv"), sep="\t", index=False)
    d_mean = float(distances.mean())
    with open(os.path.join(D, f"dlai_{r}_params.txt"), "w") as fh:
        fh.write(f"# input_file     = {trial['vcf']}\n# p1_samples     = {trial['p1']}\n# p2_sample      = {r}\n"
                 f"# p3_samples     = {trial['p3']}\n# outgroup_panel = {[trial['og']]}\n# ref_outgroup   = {trial['og']}\n"
                 f"# mean_site_w    = {1.0 if w is None else float(np.mean(w)):.6f}\n# alphabet       = {R['alph'].string}\n"
                 f"# map            = {gmap.path if gmap is not None else ''}\n")
        names = ["Background", "INT", "ILS_RD", "ILS_SD"]
        fh.write("\n[pi]\n" + "".join(f"\t{n}\t{p:.10g}\n" for n, p in zip(names, pi4)))
        fh.write(f"\n[rate_matrix_Q]\t# d_mean={d_mean:.1f} bp\nfrom\\to\t" + "\t".join(names) + "\n")
        fh.write("".join(f"{n}\t" + "\t".join(f"{Q4[i, j]:.10g}" for j in range(4)) + "\n" for i, n in enumerate(names)))
        fh.write("\n[anchor_theta_3state]\t# O->I O->L I->O I->L L->O L->I (log rates)\n"
                 + " ".join(f"{x:.10g}" for x in theta) + "\n")
    _emission_png(os.path.join(D, f"dlai_{r}_emission_by_state.png"), E)
    occ_bp = (gamma @ hmm.bp_spans(pos)) / max(hmm.bp_spans(pos).sum(), 1e-300)
    occ_site = gamma.mean(axis=1)
    sw = np.ones(len(pos)) if w is None else w
    summary = dict(p2=r, n_sites=int(len(pos)), mean_site_w=float(np.mean(sw)), frac_w_lt1=float(np.mean(sw < 0.999)),
                   alphabet=R["alph"].string, gt_pi_int=float(R["stats"]["pi_int"]), gt_prop_ils=float(R["stats"]["prop_ils"]),
                   pi_other=float(pi4[0]), pi_int=float(pi4[1]), pi_ils=float(pi4[2] + pi4[3]),
                   occ_bp_int=float(occ_bp[1]), occ_bp_ils=float(occ_bp[2]),
                   occ_site_int=float(occ_site[1]), occ_site_ils=float(occ_site[2]),
                   n_int_tracts=int((tr.state == 1).sum()),
                   int_bp=int((tr.loc[tr.state == 1, "end"] - tr.loc[tr.state == 1, "start"] + 1).sum()),
                   runtime_s=round(time.time() - t0, 1))
    log(f"Done ({r}, {summary['runtime_s']}s).")
    tr = tr.copy(); tr.insert(0, "p2", r)
    loglr = (le3[1] - le3[0]) if want_loglr else None
    return dict(p2=r, summary=summary, tracts=tr, sites=sites, loglr=loglr, theta=theta, track=(pos, gamma[1]),
                log=log.buf.getvalue())


def _emission_png(path, E):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mu, labels = E["mu"], E["labels"]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels) + 3), 5))
    x = np.arange(len(labels)); bw = 0.8 / 3
    for s, (sn, col) in enumerate(zip(("Other", "INT", "ILS"), ("grey", "tomato", "steelblue"))):
        ax.bar(x + (s - 1) * bw, mu[s], bw, label=sn, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(code | state)"); ax.set_xlabel("site code (recipient letter, P1 bin, P3 bin)")
    ax.set_title("Emission by state"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _track_png(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.2 * len(results)), squeeze=False, sharex=True)
    for ax, res in zip(axes[:, 0], results):
        pos, p = res["track"]
        if len(pos) > 8000:                       # drawn from 4,000 bins (max per bin keeps every peak): the
            b = np.linspace(pos[0], pos[-1] + 1, 4001)   # full 1e5-1e6-point polygons took ~5 s per panel to render
            k = np.clip(np.searchsorted(b, pos, side="right") - 1, 0, 3999)
            mx = np.full(4000, np.nan); np.fmax.at(mx, k, p)
            ok = np.isfinite(mx); pos, p = 0.5 * (b[:-1] + b[1:])[ok], mx[ok]
        ax.fill_between(pos, 0, p, color="tomato", lw=0, alpha=0.85)
        ax.set_ylabel(f"{res['p2']}\nP(INT)", fontsize=8); ax.axhline(0.5, color="k", ls="--", lw=0.6)
    axes[-1, 0].set_xlabel("Position (bp)"); axes[0, 0].set_title("INT posterior per recipient")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _warm_up():
    """Import matplotlib and compile the forward-backward once in the parent: forked workers inherit both instead of
    each paying for them (seconds per worker on a network file system)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    Q = np.array([[-2e-3, 1e-3, 1e-3], [1e-3, -2e-3, 1e-3], [1e-3, 1e-3, -2e-3]])
    hmm.forward_backward(np.log(np.full((3, 3), 1 / 3)), np.array([1, 10, 10]), Q, np.full(3, 1 / 3))


def _task(job):
    try:
        return decode_recipient(*job)
    except Exception as exc:
        import traceback
        return dict(p2=job[0], error=f"{type(exc).__name__}: {exc}", log=traceback.format_exc())


# --------------------------------------------------------------------------------------------- one decode
def decode_pass(D, gmap, weights, diag, want_loglr, n_workers, log=print):
    """All recipients into D. weights: {recipient: per-site weights} or None."""
    trial = _G["trial"]; p2 = trial["p2"]
    os.makedirs(D, exist_ok=True)
    pooled = None
    if len(p2) >= POOL_MIN:
        os.makedirs(os.path.join(D, "own"), exist_ok=True)
        tables = []
        for r in p2:
            R = _G["rec"][r]
            raw, t3 = em.training_counts(R["code"], R["nc"], R["lab"]["ws"], R["lab"]["ws4"],
                                         None if weights is None else weights[r])
            E = em.fit(R["nc"], raw, t3, copy_number=False, log=lambda *x: None)
            f = os.path.join(D, "own", f"mu_{r}.tsv")
            em.counts_table(R["nc"], E["cnt"] - 0.5, E["t3"], E["mu"]).to_csv(f, sep="\t", index=False)
            tables.append(pd.read_csv(f, sep="\t"))
        # the pooled counts are built from the tables as written and read back (pandas' float parser), and the decode
        # uses the pooled table as read back: the published runs did exactly this, so the counts match them to the bit
        f = os.path.join(D, "pooled_counts.tsv")
        em.pool_tables(tables).reset_index().to_csv(f, sep="\t", index=False)
        pooled = pd.read_csv(f, sep="\t").set_index("code")
        log(f"[pool] summed training counts over {len(p2)} recipients: INT {pooled.n_INT.sum():.0f}, "
            f"ILS {pooled.n_ILS.sum():.0f}, Other {pooled.n_Other.sum():.0f}")
    diag_pre = None
    if diag:
        os.makedirs(os.path.join(D, "diag"), exist_ok=True); diag_pre = os.path.join(D, "diag", "dlai")
    jobs = [(r, D, gmap, None if weights is None else weights[r], pooled, diag_pre, want_loglr) for r in p2]
    n_par = max(1, min(int(n_workers), len(p2)))
    results = []
    if n_par > 1:
        with ProcessPoolExecutor(max_workers=n_par) as ex:
            futs = [ex.submit(_task, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
    else:
        results = [_task(j) for j in jobs]
    for res in results:
        with open(os.path.join(D, f"dlai_{res['p2']}_run.log"), "w") as fh:
            fh.write(res["log"])
    bad = [res for res in results if res.get("error")]
    for res in bad:
        log(f"[decode] ERROR in recipient {res['p2']}: {res['error']}")
    ok = sorted([res for res in results if not res.get("error")], key=lambda x: p2.index(x["p2"]))
    if not ok:
        raise SystemExit("every recipient failed; see the run logs")
    pd.DataFrame([res["summary"] for res in ok]).to_csv(os.path.join(D, "dlai_multi_summary.tsv"), sep="\t", index=False)
    pd.concat([res["tracts"] for res in ok], ignore_index=True).to_csv(os.path.join(D, "dlai_all_tracts.csv"), index=False)
    pd.concat([res["sites"].assign(p2=res["p2"])[["p2"] + list(res["sites"].columns)] for res in ok],
              ignore_index=True).to_csv(os.path.join(D, "dlai_all_sites.csv.gz"), index=False, compression="gzip")
    _track_png(os.path.join(D, "dlai_p_int_by_p2.png"), ok)
    for res in ok:
        s = res["summary"]
        log(f"  {s['p2']:<14} sites {s['n_sites']:>9,}  pi_INT {s['pi_int']:.4f}  occ_bp_INT {s['occ_bp_int']:.4f}  "
            f"INT tracts {s['n_int_tracts']:>6,}  INT bp {s['int_bp']:>12,}  {s['runtime_s']:>7.1f}s")
    return ok, bad


# ------------------------------------------------------------------------------------------------- the run
OUT_FILES = ("*_tracts.csv", "*_sites.csv", "*_sites.csv.gz", "dlai_*", "mu_*.tsv", "pooled_counts.tsv",
             "diagnostics*.tsv", "gate.tsv", "gene_trees.txt")


def clear(D):
    import glob
    for d in ("diag", "own", "unweighted", "weights"):
        shutil.rmtree(os.path.join(D, d), ignore_errors=True)
    for pat in OUT_FILES:
        for f in glob.glob(os.path.join(D, pat)):
            os.remove(f)


def run(a):
    t0 = time.time()
    tick = [t0]

    def stage(name):
        now = time.time(); print(f"[time] {name}: {now - tick[0]:.0f} s", flush=True); tick[0] = now
    D = os.path.join(os.path.abspath(a.out_dir), "dlai_v2"); U = os.path.join(D, "unweighted")
    p2_names = read_samples(a.p2)
    if all(os.path.exists(os.path.join(D, f"{r}_tracts.csv")) for r in p2_names) and not a.force:
        print(f"{D}: output exists (use --force to rerun)"); return 0
    clear(D); os.makedirs(D, exist_ok=True)
    gmap = GeneticMap(a.genetic_map) if a.genetic_map else None
    print(f"dlai decode: {'TD: cM weights + map transitions' if gmap else 'Db: bp weights, no map'}", flush=True)
    _warm_up()
    T = prepare(a)
    T["vcf"] = os.path.abspath(a.vcf)
    _G.clear(); _G.update(trial=T, rec=T.pop("rec"))
    stage("prepare (VCF, sites, labels)")
    p1, p2, p3, og = T["p1"], T["p2"], T["p3"], T["og"]
    for d in (U, D):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "gene_trees.txt"), "w") as fh:
            fh.write(f"{os.path.abspath(a.gene_trees)}\n")
    full = a.diagnostics == "full"
    vg = dg.read_vcf_genotypes(a.vcf, list(p1) + p2 + list(p3) + [og], getattr(a, "contig", None)) \
        if a.diagnostics != "off" else None
    tables = lambda: {r: _G["rec"][r]["table"].assign(**({"bp_acc": _G["rec"][r]["lab"]["gt"]["bp_acc"]}
                                                          if "bp_acc" in _G["rec"][r]["lab"]["gt"] else {})) for r in p2}
    own_counts = lambda Dx: {r: pd.read_csv(os.path.join(Dx, "own", f"mu_{r}.tsv") if len(p2) >= POOL_MIN
                                            else os.path.join(Dx, f"mu_{r}.tsv"), sep="\t").set_index("code") for r in p2}
    theta4 = lambda res: {x["p2"]: np.array([float(f"{v:.4f}") for v in x["theta"]]) for x in res}
    # ---- decode 1: unweighted, no map
    print(f"=== decode 1/2: unweighted, no map -> {U}", flush=True)
    ok1, bad1 = decode_pass(U, None, None, full, True, a.n_workers)
    stage("decode 1")
    if a.diagnostics != "off":
        dg.recipient_table(U, p2, theta4(ok1), tables(), own_counts(U))
        if full:
            dg.trial_diagnostics(U, vg, p1, p2, p3, og, tables())
        dg.gate(U, len(p2))
    # ---- design-effect weights
    W = wt.build({x["p2"]: (x["sites"].pos.values, x["loglr"], x["sites"].p_int_fe.values) for x in ok1},
                 os.path.join(D, "weights"), a.genetic_map)
    stage("diagnostics 1 + weights")
    # ---- decode 2: weighted
    print(f"=== decode 2/2: weighted ({'TD' if gmap else 'Db'}) -> {D}", flush=True)
    ok2, bad2 = decode_pass(D, gmap, W, False, False, a.n_workers)
    stage("decode 2")
    if not a.keep_unweighted:
        for x in p2:
            for f in (os.path.join(U, f"{x}_sites.csv"),):
                if os.path.exists(f):
                    os.remove(f)
        if os.path.exists(os.path.join(U, "dlai_all_sites.csv.gz")):
            os.remove(os.path.join(U, "dlai_all_sites.csv.gz"))
    if a.diagnostics != "off":
        dg.recipient_table(D, p2, theta4(ok2), tables(), own_counts(D))
        dg.donor_private_per_tract(D, vg, p1, p2, p3, og)
        dg.adopt_unweighted(D, U)
        dg.gate(D, len(p2))
    for f in [os.path.join(D, f"{x}_sites.csv") for x in p2]:
        if os.path.exists(f):
            import gzip
            with open(f, "rb") as src, gzip.open(f + ".gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            os.remove(f)
    stage("diagnostics 2 + outputs")
    failed = sorted({x["p2"] for x in bad1 + bad2})
    diag_bad = sorted(__import__("glob").glob(os.path.join(U, "diag", "dlai_*_FAILED")))
    print(f"dlai decode done: {len(p2)} recipient(s) -> {D} ({time.time() - t0:.0f}s)", flush=True)
    if failed or diag_bad:
        print(f"FAILED: recipients {', '.join(failed) or '-'}; in-core diagnostics {len(diag_bad)}", file=sys.stderr)
        return 1
    return 0
