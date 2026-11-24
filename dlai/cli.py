"""dlai command line.

  dlai decode  --vcf V --p1 F --p2 F --p3 F --outgroup F --gene-trees G --out-dir O [--genetic-map M] [--mask B] ...
  dlai rnet    ...   (genome-wide structure test; see dlai rnet --help)

Sample files: whitespace-separated names. P1 = sister (non-admixed reference) panel, P2 = recipients (each decoded),
P3 = donor panel, outgroup = first name in its file (ancestral allele = REF; sites where it is not homozygous REF are
dropped). Gene trees: see dlai/genetrees.py (per-trio calls or voted tables; core_final/genetrees/ builds them)."""
import argparse
import os
import sys

# One thread per process: the per-site work is many small matrix operations; threaded BLAS only oversubscribes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _decode_parser(sp):
    p = sp.add_parser("decode", help="local ancestry (Background / INT / ILS) along one contig, every recipient",
                      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, h in (("vcf", "VCF (bgzipped or plain); biallelic SNPs are used. One contig, or see --contig"), ("p1", "sister (P1) panel sample file"),
                 ("p2", "recipient (P2) sample file"), ("p3", "donor (P3) panel sample file"),
                 ("outgroup", "outgroup sample file (first name used)"),
                 ("gene-trees", "gene-tree input: per-trio calls or voted tables (TSV[.gz]), or a directory of "
                                "dlai_<recipient>_gt_topo_*.tsv tables")):
        p.add_argument("--" + k, required=True, help=h)
    p.add_argument("--out-dir", required=True, help="outputs go to <out-dir>/dlai_v2/")
    p.add_argument("--contig", help="the contig to decode, for a VCF holding several (default: the VCF's only contig)")
    p.add_argument("--genetic-map", help="PLINK map (chrom id cM bp) of the contig: weights in cM, transitions and "
                                         "anchor run lengths in map-effective bp (TD). Without it: bp (Db)")
    p.add_argument("--mask", help="BED (0-based half-open) of regions to EXCLUDE: their SNPs are dropped, their bp are "
                                  "not counted in gene-tree windows, and they are NA rows in the tracts")
    p.add_argument("--max-site-gap", type=int, default=100000,
                   help="SNP-free stretches longer than this (bp) are treated like masked sequence (default 100000)")
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                   help="recipients decoded in parallel (default $SLURM_CPUS_PER_TASK or 4)")
    p.add_argument("--diagnostics", choices=["off", "basic", "full"], default="full",
                   help="full (default): diagnostics.tsv + model refits + diagnostics_trial.tsv; basic: diagnostics.tsv "
                        "and gate.tsv; off: none")
    p.add_argument("--keep-windows", action="store_true",
                   help="use the gene-tree windows as given (default: consecutive windows with the same call are merged)")
    p.add_argument("--keep-unweighted", action="store_true", help="keep the first decode's per-site files")
    p.add_argument("--force", action="store_true", help="rerun even if the outputs exist")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dlai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    _decode_parser(sp)
    from . import rnet
    rnet.add_parser(sp)
    a = ap.parse_args(argv)
    if a.cmd == "decode":
        from .decode import run
        return run(a)
    return rnet.run(a)


if __name__ == "__main__":
    sys.exit(main())
