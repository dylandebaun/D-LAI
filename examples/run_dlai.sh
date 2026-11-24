#!/usr/bin/env bash
#SBATCH --job-name=dlai
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
# core_final production run, one trial per array task: sticcs gene trees, then dlai decode. Same list format as
# final_run/code/launch/run_dlai.sh (lists/dlai_<set>.tsv from code/launch/dlai_design.py), so the existing design lists
# work unchanged. Line i:  set id rel_out vcf p1 p2 p3 og map mask K mode   (map / mask '-' = none; K = sticcs pairs per
# recipient, 'all' = every pair; mode TD needs a map, Db must not have one).
#   sbatch --array=1-N%M examples/run_dlai.sh <list> [<out root>]      (outputs: <out root>/<rel_out>/, default $PWD)
# Gene trees: sticcs_trees.py --vote, i.e. voted per-recipient tables, identical to the production vote_merge tables
# (per-trio rows grow as recipients x pairs x trees: ~1e8 rows for 84 recipients x 50 pairs). A finished trial
# (<out>/done) is skipped.
set -u
LIST=${1:?list}; ROOT=${2:-$PWD}
CF=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
PY=${DLAI_PY:-/home/ddebaun/nas5/miniconda3/envs/introg/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
IFS=$'\t' read -r SET ID REL VCF P1 P2 P3 OG MAP MASK K MODE < <(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$LIST")
[ -n "${MODE:-}" ] || { echo "bad line ${SLURM_ARRAY_TASK_ID:-1} of $LIST"; exit 2; }
case "$MODE" in
  TD) [ "$MAP" != - ] || { echo "TD row without a map"; exit 2; } ;;
  Db) [ "$MAP" = - ] || { echo "Db row with a map"; exit 2; } ;;
  *)  echo "unknown mode $MODE"; exit 2 ;;
esac
NW=${SLURM_CPUS_PER_TASK:-4}; O=$ROOT/$REL
[ -f "$O/done" ] && { echo "done already: $O"; exit 0; }
mkdir -p "$O"; export TMPDIR=${TMPDIR:-/tmp}/dlai_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
t0=$(date +%s)
KB=$([ "$K" = all ] && echo 100000 || echo "$K")
GT=$O/gene_trees.tsv.gz
if [ ! -s "$GT" ]; then
  $PY "$CF/genetrees/sticcs_trees.py" --vcf "$VCF" --p1 "$P1" --p2 "$P2" --p3 "$P3" --outgroup "$OG" --out "$GT" \
    --combos "$KB" --vote --n-workers "$NW" > "$O/sticcs.log" 2>&1 || { echo "sticcs FAILED"; tail -20 "$O/sticcs.log"; exit 1; }
fi
t1=$(date +%s)
CMD=($PY "$CF/bin/dlai" decode --vcf "$VCF" --p1 "$P1" --p2 "$P2" --p3 "$P3" --outgroup "$OG" --gene-trees "$GT"
     --out-dir "$O" --n-workers "$NW" --force)
[ "$MODE" = TD ] && CMD+=(--genetic-map "$MAP")
[ "$MASK" = - ] || CMD+=(--mask "$MASK")
{ echo "dlai run: set $SET id $ID mode $MODE K $K host $(hostname) start $(date -u +%FT%TZ)"
  echo "command: ${CMD[*]}"; echo "core_final md5:"; md5sum "$CF"/bin/dlai "$CF"/dlai/*.py "$CF"/genetrees/*.py
  echo "vcf md5: $(md5sum "$VCF" | cut -d' ' -f1)"; } > "$O/settings.txt"
"${CMD[@]}" > "$O/run.log" 2>&1; rc=$?
t2=$(date +%s)
echo "elapsed_s sticcs $(( t1 - t0 )) decode $(( t2 - t1 )) rc $rc" >> "$O/settings.txt"
[ $rc -eq 0 ] || { echo "dlai FAILED rc=$rc"; tail -20 "$O/run.log"; exit $rc; }
touch "$O/done"; echo "ok: sticcs $(( t1 - t0 )) s, decode $(( t2 - t1 )) s"
