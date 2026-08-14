# D-LAI: A D-statistic local ancestry inference model

D-LAI is a method designed to infer introgressed ancestry tracts from diploid variant call datasets. The model takes as input an unphased VCF with designated sister (P1), hybrid (P2), donor (P3), and outgroup (O) individuals, with as few as one representative per population. 

## Requirements
- **Python 3** with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
  `seaborn`.
- **IQ-TREE** (used to infer gene trees). Point the model at it with
  `--iqtree_bin /path/to/iqtree` (default `iqtree`).

## Usage

```bash
dlai --vcf  $VCF
    --p1 $P1-INDIV1 $P1-INDIV2 --p3 $P1-INDIV1 --p2 $P2-INDIV1 --outgroup $OUT-INDIV1 \
    --adaptive_anchor \
    --n_workers 4 \
    --out_prefix results
```

Full option list: `python ctmc_dm_SUBMISSION.py --help`.

## Outputs

| File | Contents |
|------|----------|
| `_sites.csv` | per-site posteriors (`p_other_fe`, `p_int_fe`, `p_ils_fe`) and decoded state `y_hat_fe`; oracle columns `*_sup` when `--tract_file` is given |
| `_tracts.csv` | predicted tracts: `start,end,state,n_sites,mean_p_int,mean_p_int_bp`; plus `propintro,propils` (truth overlap) when `--tract_file` is given |
| `_params.txt` | fitted transition matrix, generator Q, emission matrices, and π |
| `_results.png` | confusion matrices, INT-posterior histograms, emission heatmaps |
| `_gt_topo_*.tsv` | cached gene-tree topologies |


## Citation











