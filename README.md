# D-LAI

D-LAI labels every position of a genome, for each recipient genome, as **Background**, **introgression (INT)** or
**incomplete lineage sorting (ILS)**. It works from a VCF and gene trees that you supply. The model is a four-state
continuous-time hidden Markov model:

- States: Background, INT, ILS_RD (recipient-donor ILS) and ILS_SD (sister-donor ILS). ILS is reported as ILS_RD + ILS_SD.
- Training: the emissions are trained on gene-tree windows labelled from their topologies. No simulation truth is used.
- Transitions: the generator is anchored on the run lengths of those labels. It is not fitted to the data.

On top of the decode, `dlai` adds:
- design-effect evidence weights, from a first unweighted decode;
- an optional genetic map, so transitions run in genetic distance;
- a copy-number annotation, P(two introgressed copies | INT).

`dlai rnet` computes the genome-wide R_net structure test.

This is the production package (`final_run/core_final/`, 2026-11-24). It gives the same calls and outputs as the
research code in `final_run/core/`, which is unchanged, and runs 3-10x faster.

Contents: [Install](#install) | [Quick start](#quick-start) | [Inputs](#inputs) | [Gene trees](#gene-trees) |
[dlai decode](#dlai-decode) | [Outputs](#outputs-of-dlai-decode) | [dlai rnet](#dlai-rnet) |
[Gene-tree builders](#gene-tree-builders) | [Running on SLURM](#running-on-slurm) | [Validation](#validation) |
[Troubleshooting](#troubleshooting)

```
core_final/
  bin/dlai                 entry point: dlai decode | dlai rnet
  dlai/                    the method (Python package)
    cli.py                 command line
    io.py                  VCF reader (bcftools, or Python fallback), mask, unsupported spans, genetic map
    genetrees.py           gene-tree input: per-trio calls (voted here) or voted tables
    sites.py               polarisation, site patterns, informative-site filter, emission codes
    labels.py              gene-tree window labels, transition anchor (three- and four-state)
    emission.py            training counts, pooling over recipients, per-state emission, copy-number split
    hmm.py                 generators, forward-backward (numba), copy-number generator, tracts
    decode.py              dlai decode: the two decodes and the evidence weights
    weights.py             design-effect evidence weights
    diagnostics.py         diagnostics.tsv, gate.tsv, in-core refits, diagnostics_trial.tsv
    rnet.py                dlai rnet
  genetrees/               gene-tree builders (separate programs; dlai never runs them)
    sticcs_trees.py        stacked sticcs sub-ARGs (+ vendored external/sticcs)
    iqtree_trees.py        per-window 4-taxon IQ-TREE trees
  examples/run_dlai.sh     SLURM array launcher (final_run list format): sticcs trees, then dlai decode
```


## Install

You need Python 3.12 with numpy, scipy, pandas, scikit-allel, matplotlib and numba. The `introg` conda environment
on the cluster has all of them:

```
PY=/home/ddebaun/nas5/miniconda3/envs/introg/bin/python
$PY /nas5/ddebaun/SIMULATION_tests/final_run/core_final/bin/dlai --help
```
You can also put `core_final/bin` on `PATH` and run `dlai`; the script finds its package next to itself.

- **bcftools** is optional. dlai looks first in the Python environment's `bin/`, then on `PATH`. With it, dlai reads
  VCFs 1.4-3.5x faster. Without it, a pure-Python reader gives the same result.
- **IQ-TREE 2 or 3** is needed only by `genetrees/iqtree_trees.py`.
- **Threads:** set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` (`dlai` sets these itself). The work is many
  small matrix operations, and a threaded BLAS in every worker only oversubscribes the node.


## Quick start

```
D=/nas5/ddebaun/SIMULATION_tests/final_run/core_final
PY=/home/ddebaun/nas5/miniconda3/envs/introg/bin/python

# 1. gene trees: sticcs, 50 (sister, donor) pairs per recipient, voted per recipient
$PY $D/genetrees/sticcs_trees.py --vcf chr1.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt \
    --outgroup og.txt --out trees.chr1.tsv.gz --combos 50 --vote --n-workers 16

# 2. decode every recipient (with a genetic map: TD; without one: Db)
$PY $D/bin/dlai decode --vcf chr1.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt --outgroup og.txt \
    --gene-trees trees.chr1.tsv.gz --genetic-map chr1.plink.map --out-dir run_chr1 --n-workers 16

# 3. calls: run_chr1/dlai_v2/<recipient>_tracts.csv (state 1 = INT), posteriors: <recipient>_sites.csv.gz

# 4. genome-wide structure test from one-genome combinations (needs per-trio trees: no --vote)
$PY $D/genetrees/sticcs_trees.py --vcf chr1.vcf.gz --p1 sisters.txt --p2 recipients.txt --p3 donors.txt \
    --outgroup og.txt --out trio.chr1.tsv.gz --combos 100000 --n-workers 16     # every (sister, donor) pair
# ... the same for every chromosome, then:
$PY $D/bin/dlai rnet --vcf chr*.vcf.gz --gene-trees trio.chr*.tsv.gz --combos combos.txt --outgroup og.txt \
    --out rnet_genome --workers 16
```
(`chr*` expands in the same order for the VCFs and the gene trees, chr1 chr10 chr11 ...: the two lists only need to
match each other.)


## Inputs

**Sample files.** One file per role, holding sample names separated by whitespace or newlines:

| role | flag | meaning |
|---|---|---|
| P1 | `--p1` | sister panel: non-admixed relatives of the recipients. The first name is the *reference sister*. |
| P2 | `--p2` | recipients: every one is decoded separately |
| P3 | `--p3` | donor panel. The first name is the *reference donor*. |
| outgroup | `--outgroup` | one outgroup genome (the first name is used); the ancestral allele is REF |

**VCF.** The VCF can be bgzipped or plain.
- Only biallelic SNPs are used: single-base A/C/G/T REF and ALT. Other records are skipped and counted in the log.
- Records sharing a position are all dropped.
- Genotypes are unphased diploid calls. A genotype with any uncalled allele counts as uncalled.
- Only sites where the outgroup is called homozygous REF are used, so REF must be the ancestral allele.
- One contig per run: either the VCF's only contig, or `--contig` (decode) / `--contigs` (rnet) for a whole-genome VCF.
  A whole-genome VCF should be indexed (`tabix -p vcf`). The diagnostics' second read needs the index; the main reader
  streams the file when there is none.

**Mask** (`--mask`, optional). A BED file (0-based half-open, optionally gzipped) of regions to **exclude**:
- SNPs inside the regions are dropped;
- masked bp do not count towards gene-tree window lengths;
- masked regions become NA rows (state -1) in the tracts.

Only the BED rows of the decoded contig are used when the BED names it. SNP-free stretches longer than
`--max-site-gap` (100 kb) are treated the same way.

**Genetic map** (`--genetic-map`, optional). A PLINK map of the contig, with whitespace-separated columns
`chrom id cM bp`. With a map:
- the evidence weights use genetic distance (cM);
- the weighted decode measures transitions and anchor run lengths in map-effective bp ("TD").

Without a map, both use bp ("Db").


## Gene trees

dlai does not infer gene trees. It reads them in one of two forms: a tab-separated file (optionally gzipped) with a
header line, 1-based inclusive coordinates, and these columns:

| form | columns | what dlai does |
|---|---|---|
| per-trio calls | `chrom start end recipient sister donor topology` | votes them per recipient (below) |
| voted tables | `chrom start end recipient topology` | uses them as they are |

- **topology:** `T1` = ((sister, recipient), donor), `T2` = ((recipient, donor), sister),
  `T3` = ((sister, donor), recipient), all rooted with the outgroup. `NA` or an empty field means unresolved.
- **chrom column:** it can be left out for a single-contig run.
- **Legacy input:** a *directory* of `dlai_<recipient>_gt_topo_*_w25.tsv` tables, as written by the `final_run/core`
  builders and found in production run directories, is also accepted as voted tables.

**Vote of per-trio calls.** Per recipient:
1. The trios' intervals are stacked on the union of their breakpoints.
2. Each segment takes the plurality topology over the trios.
3. A tie takes the reference trio's call (first `--p1` x first `--p3` sample) when that call is among the tied
   topologies. Otherwise it is drawn at random, with a fixed seed per recipient. This is the rule of the sticcs
   builder used for the production runs.

**Windows.** Neighbouring windows with the same call are merged into one window. This is the production `vote_merge`
setting. `--keep-windows` uses the windows exactly as given; decode IQ-TREE trees this way (see below).

**Which form to use.** A per-trio file grows as recipients x pairs x trees: about 10^8 rows for 84 recipients x 50
pairs. For large panels, write voted tables (`sticcs_trees.py --vote`), which are identical to the production tables.
`dlai rnet` needs per-trio calls.


## dlai decode

What a run does:
1. Read the VCF once: the wanted samples, biallelic SNPs, mask applied. Polarise by the outgroup and read the gene trees.
2. Per recipient: the decoded sites, their emission codes and the gene-tree window labels.
   - **Decoded sites:** polarised, recipient called, and informative for at least one (P1 sample, P3 sample) pair; or
     the P1 panel uncalled while the donor panel is called.
   - **Labels:** T1 and T3 windows are Background. T2 windows are ranked by rank(f_d) + rank(donor-private sites).
     The top-ranked T2 windows are labelled INT until their summed bp is closest to the T2 excess (bp of T2 windows
     minus bp of T3 windows). INT windows with fewer than 2 donor-private sites become ILS. The rest are ILS.
3. **Decode 1** goes to `dlai_v2/unweighted/`. It is unweighted, without the map.
   - Emission: each site's code is recipient letter x P1-panel frequency bin x P3-panel frequency bin.
   - With 2 or more recipients, the emission is fitted from the counts pooled over all recipients, each recipient's
     sites counted once.
   - Four-state generator from the label run lengths, then forward-backward, posteriors and tracts.
   - Its diagnostics are the reported ones.
4. **Evidence weights** go to `dlai_v2/weights/`. Each site's weight is w = 1 / (1 + sum of rho(d) over its 200
   neighbours on each side), with rho = the correlation of the decode-1 log-likelihood-ratio residuals within runs.
5. **Decode 2** goes to `dlai_v2/`: the weights scale the training counts and log-emissions. These are the calls.
6. **Copy number:** INT, ILS_RD and ILS_SD are split into one- and two-copy states in a second forward-backward pass.
   It gives `p_int2_fe` and `two_copy_share`, and leaves the calls and P(INT) unchanged. It is skipped for a recipient
   whose fitted two-copy share is below 0.02 or that has fewer than 200 INT training sites.

Recipients are decoded in parallel (`--n-workers` processes).

### Usage (`dlai decode --help`)

```
usage: dlai decode [-h] --vcf VCF --p1 P1 --p2 P2 --p3 P3 --outgroup OUTGROUP --gene-trees GENE_TREES --out-dir
                   OUT_DIR [--contig CONTIG] [--genetic-map GENETIC_MAP] [--mask MASK] [--max-site-gap MAX_SITE_GAP]
                   [--n-workers N_WORKERS] [--diagnostics {off,basic,full}] [--keep-windows] [--keep-unweighted]
                   [--force]

dlai command line.

  dlai decode  --vcf V --p1 F --p2 F --p3 F --outgroup F --gene-trees G --out-dir O [--genetic-map M] [--mask B] ...
  dlai rnet    ...   (genome-wide structure test; see dlai rnet --help)

Sample files: whitespace-separated names. P1 = sister (non-admixed reference) panel, P2 = recipients (each decoded),
P3 = donor panel, outgroup = first name in its file (ancestral allele = REF; sites where it is not homozygous REF are
dropped). Gene trees: see dlai/genetrees.py (per-trio calls or voted tables; core_final/genetrees/ builds them).

options:
  -h, --help            show this help message and exit
  --vcf VCF             VCF (bgzipped or plain); biallelic SNPs are used. One contig, or see --contig
  --p1 P1               sister (P1) panel sample file
  --p2 P2               recipient (P2) sample file
  --p3 P3               donor (P3) panel sample file
  --outgroup OUTGROUP   outgroup sample file (first name used)
  --gene-trees GENE_TREES
                        gene-tree input: per-trio calls or voted tables (TSV[.gz]), or a directory of
                        dlai_<recipient>_gt_topo_*.tsv tables
  --out-dir OUT_DIR     outputs go to <out-dir>/dlai_v2/
  --contig CONTIG       the contig to decode, for a VCF holding several (default: the VCF's only contig)
  --genetic-map GENETIC_MAP
                        PLINK map (chrom id cM bp) of the contig: weights in cM, transitions and anchor run lengths
                        in map-effective bp (TD). Without it: bp (Db)
  --mask MASK           BED (0-based half-open) of regions to EXCLUDE: their SNPs are dropped, their bp are not
                        counted in gene-tree windows, and they are NA rows in the tracts
  --max-site-gap MAX_SITE_GAP
                        SNP-free stretches longer than this (bp) are treated like masked sequence (default 100000)
  --n-workers N_WORKERS
                        recipients decoded in parallel (default $SLURM_CPUS_PER_TASK or 4)
  --diagnostics {off,basic,full}
                        full (default): diagnostics.tsv + model refits + diagnostics_trial.tsv; basic:
                        diagnostics.tsv and gate.tsv; off: none
  --keep-windows        use the gene-tree windows as given (default: consecutive windows with the same call are
                        merged)
  --keep-unweighted     keep the first decode's per-site files
  --force               rerun even if the outputs exist
```


## Outputs of dlai decode

Everything goes to `<out-dir>/dlai_v2/`. The layout and file names are those of the production runs, so the scoring
and figure code reads them unchanged.

| file | content |
|---|---|
| `<r>_tracts.csv` | **the calls**: one row per tract of the hard call |
| `<r>_sites.csv.gz` | per-site posteriors |
| `dlai_all_tracts.csv`, `dlai_all_sites.csv.gz` | the same, every recipient (column `p2`) |
| `dlai_multi_summary.tsv` | one row per recipient |
| `diagnostics.tsv`, `diagnostics_trial.tsv`, `gate.tsv` | diagnostics (from decode 1; see below) |
| `diagnostics_weighted.tsv` | decode 2's own per-recipient diagnostics |
| `mu_<r>.tsv` | fitted emission per state (`mu_*`) and training counts (`n_*`; pooled counts when pooled) |
| `own/mu_<r>.tsv`, `pooled_counts.tsv` | each recipient's own counts and their cohort sum (pooled runs only) |
| `mu_<r>.int2.tsv` | one- / two-copy INT emission (when the copy-number split is fitted) |
| `dlai_<r>_gt_topo_<sister>_<donor>_w25.tsv` | the gene-tree windows used, with `n_snps` and the per-window distances `med_d12 / d23 / d13 / d2o` |
| `dlai_<r>_window_bp.tsv` | supported bp per window (only when there are masked or unsupported spans) |
| `dlai_<r>_params.txt` | input file, samples, stationary pi and generator Q of the four-state decode, three-state anchor |
| `dlai_<r>_run.log`, `dlai_<r>_emission_by_state.png`, `dlai_p_int_by_p2.png` | per-recipient log, emission plot, P(INT) track |
| `weights/<r>.tsv`, `weights/{summary,rho}.tsv`, `weights/rule.txt` | the evidence weights and their correlation curve |
| `unweighted/` | decode 1: the same files; its per-site files are removed unless `--keep-unweighted`. `unweighted/diag/` holds the in-core refits |
| `gene_trees.txt` | the gene-tree input used |

**`<r>_tracts.csv`** columns:

| column | meaning |
|---|---|
| `start`, `end` | 1-based inclusive. Boundaries are placed inside the gap between two decoded sites, at the base where the posterior of the left call falls below that of the right one |
| `state` | 0 = Background (Other), 1 = INT, 2 = ILS, -1 = NA (masked or unsupported span) |
| `n_sites` | decoded sites in the tract |
| `site_i0`, `site_i1` | first and last site index in `<r>_sites.csv` |
| `mean_p_int` | mean site P(INT) |
| `mean_p_int_bp` | mean P(INT) with each site weighted by its bp |
| `two_copy_share` | P(two copies \| INT), posterior-weighted (when the copy-number split is fitted) |
| `n_donor_private`, `donor_private_per_kb` | sites where the recipient carries a derived allele that no P1-panel member has and the donor panel has; the only codes that separate INT from ILS, so a low count marks a weakly determined call. Final decode only |

**`<r>_sites.csv.gz`** has one row per decoded site:
- `pos`;
- the posteriors `p_other_fe`, `p_int_fe`, `p_ils_fe` (= `p_ils_rdt` + `p_ils_sdt`);
- the hard call `y_hat_fe` (0 / 1 / 2);
- `p_int2_fe` = P(INT and two copies), when the copy-number split is fitted.

**`dlai_multi_summary.tsv`** has one row per recipient:

| column | meaning |
|---|---|
| `n_sites` | decoded sites |
| `mean_site_w`, `frac_w_lt1` | mean evidence weight, and the fraction of sites below full weight |
| `gt_pi_int`, `gt_prop_ils` | gene-tree T2 excess and T3 fraction, counted in windows |
| `pi_other`, `pi_int`, `pi_ils` | stationary distribution of the generator |
| `occ_bp_*`, `occ_site_*` | posterior occupancy, bp-weighted and site-weighted |
| `n_int_tracts`, `int_bp` | INT tracts and their summed bp |
| `runtime_s` | time for this recipient |

**Diagnostics.** These are computed from decode 1 and never change a call.

`diagnostics.tsv` has one row per recipient, with these groups of columns:
- topology fractions and T2 excess, in windows and in bp;
- T2/T3 linkage ratio and gene-tree distance ratios;
- anchor pi and dwell times;
- het share;
- excess JS divergence INT-ILS and INT-Other;
- % called INT;
- the in-core refits (log-likelihoods of models B0 / B / Tc / To, dwell profile cost, ILSwin);
- rate ratio `rr_min`;
- divergence ratios in INT vs ILS tracts (`Q23 Q13 Q12 Q23n`);
- D, f_hom and f_d.

The other two files:
- `diagnostics_trial.tsv` is one row for the whole trial: structure ratio R and R6 with block-jackknife SE, the means
  of the per-recipient columns, `decision` (report / warn / zero) and `fast_lineage`.
- `gate.tsv` holds the structure ratio R = mean JS_INT_ILS_excess / mean JS_INT_Other_excess. It is reported only:
  `threshold` 0, `flagged` 0.

**Exit status.** 0 = every recipient decoded. 1 = some recipient failed: its error is in `dlai_<r>_run.log`, and the
other recipients' outputs are still written. A failed in-core refit leaves `unweighted/diag/dlai_<r>_FAILED` and also
exits 1, while the decode itself is unaffected.

**Resources** (production smoke trials, 16 CPUs unless noted):

| trial | recipients | decoded sites / recipient | time |
|---|---:|---:|---:|
| simulated minimal (4 CPUs) | 1 | 108k | 39 s |
| simulated hToN, map (8 CPUs) | 3 | 49k | 30 s |
| simulated normal 10/10/10, map | 10 | 146k | 211 s |
| simulated Gower84, map (32 CPUs) | 84 | ~50k | 464 s |
| NEA chr21, mask, map | 3 | ~54k | 31 s |
| CEU84 chr21, mask, map | 84 | ~31k | 342 s |
| Heliconius chr8 | 3 | ~62k | 42 s |

Memory is a few GB, dominated by the VCF genotypes of the wanted samples.


## dlai rnet

R_net measures whether the INT calls of a data set carry the signal of introgression or of deep ancestral structure.
It works from one-genome *combinations*: one recipient, one sister and one donor genome (+ the outgroup), each with
that trio's own gene trees. All chromosomes of a combination are laid end to end, and the statistic is computed once
for the genome. The genome value is the mean over combinations. Standard errors come from a 20-block genome jackknife
and, with several chromosomes, a delete-one-chromosome jackknife.

**Inputs:**
- **VCFs:** one VCF per chromosome (`--vcf chr1.vcf.gz chr2.vcf.gz ...`), or one whole-genome VCF with
  `--contigs chr1,chr2,...`.
- **`--combos` file:** one combination per line, `recipient sister donor` (whitespace-separated).
- **`--gene-trees`:** per-trio calls that hold every combination's trio on every chromosome. Give one file with a chrom
  column, or one file per chromosome in the same order as the VCFs. `sticcs_trees.py` without `--vote` writes them;
  run it with the sisters as `--p1`, the recipients as `--p2`, the donors as `--p3`, and `--combos` at least the
  number of (sister, donor) pairs, so every pair is included.
- **`--mask`** (optional): one BED for all contigs, or one per chromosome.

The supported window bp exclude the mask and SNP-free stretches longer than `--max-site-gap`, measured on the SNPs
polymorphic in the combination's four genomes.

### Usage (`dlai rnet --help`)

```
usage: dlai rnet [-h] --vcf VCF [VCF ...] [--contigs CONTIGS] --gene-trees GENE_TREES [GENE_TREES ...] --combos
                 COMBOS --outgroup OUTGROUP [--mask MASK [MASK ...]] --out OUT [--label LABEL]
                 [--max-site-gap MAX_SITE_GAP] [--workers WORKERS]

dlai rnet: genome-wide R_net structure test (dlaidraft2 "Structure diagnostic test"; port of
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
-> PREFIX.tsv (one row: R_net, SE, CI, p05 = P(t_19 >= (R_net - 0.05) / SE), ...), PREFIX.combos.tsv (per combination).

options:
  -h, --help            show this help message and exit
  --vcf VCF [VCF ...]   one VCF per chromosome, or one VCF holding several contigs together with --contigs
  --contigs CONTIGS     comma-separated contigs to read from a single multi-contig --vcf (indexed: read through the
                        index)
  --gene-trees GENE_TREES [GENE_TREES ...]
                        per-trio gene-tree calls: one file with a chrom column, or one file per VCF
  --combos COMBOS       file: recipient sister donor per line
  --outgroup OUTGROUP   outgroup sample file (first name used)
  --mask MASK [MASK ...]
                        exclusion BED(s): one for all contigs, or one per VCF
  --out OUT             output prefix
  --label LABEL         label written in the label column of both outputs
  --max-site-gap MAX_SITE_GAP
                        SNP-free stretches longer than this (bp) do not count as supported window bp (default
                        100000)
  --workers WORKERS     combinations computed in parallel (default $SLURM_CPUS_PER_TASK or 1)
```

**Outputs:**

`PREFIX.tsv` is one row for the genome:

| column | meaning |
|---|---|
| `rnet` | R_net, the mean over combinations |
| `se`, `lo`, `hi` | 20-block jackknife SE and 95% CI (t, 19 df) |
| `p05`, `p0` | P(t_19 >= (R_net - 0.05) / SE), and the same with 0 |
| `R_mean` | mean JS*(INT, ILS) / JS*(INT, Background) |
| `q_mean` | mean INT share of T2 bp |
| `chr_*` | delete-one-chromosome jackknife (several chromosomes only) |
| `rnet_pool`, `pool_*` | the statistic of the counts summed over combinations, and its jackknife |

`PREFIX.combos.tsv` has one row per combination: `combo` (recipient__sister__donor), `q`, `rnet`, `R`, `n_win`
(gene-tree windows) and `n_sites` (sites used).

Time: about 1 s per chromosome to read the VCF through bcftools, plus the statistic, which runs in seconds per
combination. Both stored checks (5 combinations on one simulated chromosome; 4 combinations on CEU84 chr21 + chr22)
take about 7 s.


## Gene-tree builders

These are separate programs. dlai only reads their output.

### sticcs (`genetrees/sticcs_trees.py`)

This is the gene-tree source of the production runs. For each recipient, K (sister, donor) pairs are taken: the
reference pair first, then a seeded shuffle. Each pair gets its own sticcs inference on the three diploids. The output
is per-trio calls, or voted tables with `--vote`, which are identical to the production `vote_merge` tables.

```
usage: sticcs_trees.py [-h] --vcf VCF --p1 P1 --p2 P2 --p3 P3 --outgroup OUTGROUP --out OUT [--combos COMBOS]
                       [--vote] [--seq-len SEQ_LEN] [--n-workers N_WORKERS] [--seed SEED] [--phased]

sticcs_trees.py -- gene-tree input for dlai from stacked sticcs sub-ARGs (separate from dlai; dlai never runs it).

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

    python sticcs_trees.py --vcf X.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt --outgroup og.txt \
        --out trees.tsv.gz [--combos 64] [--vote] [--n-workers N] [--phased]
Sample files: whitespace-separated names. One contig per VCF (biallelic SNPs; ancestral = REF, outgroup hom-REF sites).

options:
  -h, --help            show this help message and exit
  --vcf VCF             VCF (one contig; biallelic SNPs used)
  --p1 P1               sister (P1) panel sample file
  --p2 P2               recipient (P2) sample file
  --p3 P3               donor (P3) panel sample file
  --outgroup OUTGROUP   outgroup sample file (first name used; ancestral = REF)
  --out OUT             output TSV (.gz)
  --combos COMBOS       (sister, donor) pairs per recipient (default 64)
  --vote                write voted per-recipient tables instead of per-trio calls
  --seq-len SEQ_LEN     end of the last tree (default: last SNP position)
  --n-workers N_WORKERS
                        worker processes (default $SLURM_CPUS_PER_TASK or 4)
  --seed SEED           pair-order and tie seed (default 1)
  --phased              use the VCF's phase: every diploid becomes two haploid sticcs samples (the VCF must be
                        phased)
```

### IQ-TREE (`genetrees/iqtree_trees.py`)

This is the old `--gene-trees iqtree` path: 25-SNP windows, K random or nested pairs, one 4-taxon IQ-TREE tree per
window and pair. Its per-trio rows reproduce the old per-window tables exactly, using the production vote (see the
docstring). **Decode these trees with `dlai decode --keep-windows`**, because the old IQ-TREE path never merged
windows.

```
usage: iqtree_trees.py [-h] --vcf VCF --p1 P1 --p2 P2 --p3 P3 --outgroup OUTGROUP --out OUT [--combos COMBOS]
                       [--pair-order {random,nested}] [--seed SEED] [--iqtree-seed IQTREE_SEED]
                       [--snp-window SNP_WINDOW] [--batch-size BATCH_SIZE] [--n-workers N_WORKERS]
                       [--iqtree-bin IQTREE_BIN]

iqtree_trees.py -- gene-tree input for dlai from per-window 4-taxon IQ-TREE trees (separate from dlai; dlai never
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
    d = pd.read_csv(out, sep="\t", keep_default_na=False)
    d.groupby(["recipient", "start", "end"], sort=False).topology.agg(vote)
(dlai votes the rows itself, with its own tie rule.) Identical trees need the production batch layout as well as its
inputs: IQ-TREE's tree for one alignment depends on the other alignments in its -S call, so recipients (--p2 order),
pairs, --snp-window, --batch-size, --iqtree-seed and the IQ-TREE binary must all match; the alignment files carry the
production names (<recipient>__<window>__<sister>__<donor>.fasta, which -S processes in sorted order).

    python iqtree_trees.py --vcf X.vcf.gz --p1 sister.txt --p2 recipients.txt --p3 donor.txt --outgroup og.txt \
        --out trees.tsv.gz [--combos 5] [--pair-order random|nested] [--seed 2] [--n-workers N]
Sample files: whitespace-separated names. One contig per VCF. Defaults are the production ones (dlai_v2.py --gene-trees
iqtree: 5 pairs, random order, pair seed 2, IQ-TREE seed 1, 25-SNP windows, 5,000 alignments per IQ-TREE call).
Decode these trees with `dlai decode --keep-windows`: the IQ-TREE path of the old core used its 25-SNP windows as given
(never merged). The gaps between windows become NA segments in dlai's vote, which carry no label, exactly like the
positions outside every window in the old core.

options:
  -h, --help            show this help message and exit
  --vcf VCF             VCF (one contig; biallelic SNPs used)
  --p1 P1               sister (P1) panel sample file
  --p2 P2               recipient (P2) sample file
  --p3 P3               donor (P3) panel sample file
  --outgroup OUTGROUP   outgroup sample file (first name used; ancestral = REF)
  --out OUT             output TSV (.gz)
  --combos COMBOS       (sister, donor) pairs per window (default 5; >= the number of pairs = every pair)
  --pair-order {random,nested}
                        random: a seeded sample of --combos pairs; nested: the reference pair first, then a seeded
                        shuffle, the first --combos used (default random)
  --seed SEED           pair-selection seed (production gt_combos_seed, default 2)
  --iqtree-seed IQTREE_SEED
                        IQ-TREE -seed (production --seed, default 1)
  --snp-window SNP_WINDOW
                        SNPs per window (default 25)
  --batch-size BATCH_SIZE
                        alignments per IQ-TREE -S call (default 5000)
  --n-workers N_WORKERS
                        worker processes (default $SLURM_CPUS_PER_TASK or 4)
  --iqtree-bin IQTREE_BIN
                        IQ-TREE 2/3 binary (default /home/ddebaun/nas5/miniconda3/envs/introg/bin/iqtree3)
```


## Running on SLURM

`examples/run_dlai.sh` is an array launcher. It reads the line format of `final_run/code/launch/run_dlai.sh` (lists
made by `code/launch/dlai_design.py`):

```
set  id  rel_out  vcf  p1  p2  p3  outgroup  map  mask  K  mode        (map / mask '-' = none; K 'all' = every pair;
                                                                         mode TD needs a map, Db must not have one)
sbatch --array=1-N%M -c 16 --mem 48G /nas5/ddebaun/SIMULATION_tests/final_run/core_final/examples/run_dlai.sh \
    lists/dlai_<set>.tsv <out root>
```
For each line, the launcher:
1. builds the sticcs trees (`--vote`);
2. runs `dlai decode`;
3. writes `<out root>/<rel_out>/settings.txt`, containing the command, the core_final md5 sums, the VCF md5 and the
   elapsed times;
4. marks the trial done.

A finished trial is skipped. For a whole genome, give one line per chromosome, or loop over `--contig` with one
indexed VCF.


## Validation

`final_run/port_check/core_final_2026-11-24/` (see `NOTES.md` there) compares `core_final` with the production core
(`final_run/core`, as run in `port_check/quartet_port/runs/new_prod`) on the 7 production smoke trials. They cover
simulated normal, minimal, hToN and Gower84, and empirical NEA chr21, CEU84 chr21 and Heliconius chr8; TD and Db;
masks; and 1-84 recipients.

**Decode:** compared file by file with `compare_tol.py`.
- Calls and tract boundaries are identical.
- Every numeric column is within 1e-9 relative when run on the same CPU type as the reference.
- On different CPU types, the copy-number columns can differ in the last digits.

**End to end:** `sticcs_trees.py --vote` followed by `dlai decode` gives the same result.

**R_net:** `dlai rnet` reproduces the stored R_net checks of `code/structure/check_rnet_gw.sh` exactly, both from
per-chromosome VCFs and from one two-contig VCF.

**Readers and builders:**
- The bcftools VCF reader is bit-identical to the Python reader on all validation VCFs.
- `iqtree_trees.py` reproduces the stored IQ-TREE tables window for window.

**Differences from `final_run/core`:**
- Gene trees are supplied; dlai no longer builds them.
- Per-trio calls are voted with the sticcs tie rule.
- No `DM_* / DLAI_* / SYM_* / GT_*` environment switches. Options that production never used are removed.
- Output changes:
  - the always-empty truth and path columns are removed from `dlai_multi_summary.tsv`;
  - `params.txt` is shorter but keeps the header lines R_net reads;
  - no `outgroup_groups.txt`;
  - final per-site files are written gzipped.
- A mask BED is filtered to the decoded contig when it names it.


## Troubleshooting

| message | cause / fix |
|---|---|
| `holds more than one contig` | give `--contig` (decode) or `--contigs` (rnet) |
| `samples not in the VCF header` | a name in a sample file is not in the VCF |
| `gene trees ...: no rows for recipient` | the gene-tree file lacks that recipient (or that trio / contig) |
| `overlapping windows for recipient` | a voted table has overlapping rows; give per-trio calls (with sister / donor) instead |
| `only N usable sites for recipient` | fewer than 100 polarised sites with the recipient called |
| `bcftools could not read ... reading it in Python` | bcftools failed on the file; the Python reader is used (same result, slower) |
| `region` errors from scikit-allel with `--contig` | the VCF needs a tabix index (`tabix -p vcf`) |
