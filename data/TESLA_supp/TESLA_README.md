# TESLA Consortium Benchmark Dataset

## Source

This data comes from the Tumor Neoantigen Selection Alliance (TESLA), a global consortium of 36+ neoantigen research groups convened by the Parker Institute for Cancer Immunotherapy (PICI) and the Cancer Research Institute (CRI).

The files were downloaded from the supplementary materials of:

> Wells, D.K., van Buuren, M.M., Dang, K.K. et al.
> "Key Parameters of Tumor Epitope Immunogenicity Revealed Through a Consortium Approach Improve Neoantigen Prediction."
> *Cell* 183(3), 818-834.e13 (2020).
> DOI: [10.1016/j.cell.2020.09.015](https://doi.org/10.1016/j.cell.2020.09.015)

## Files

### `s4_pMHC_batch_1.xlsx`

Supplementary Table S4 from Wells et al. (2020). Contains 608 peptide-MHC pairs from the first TESLA batch (6 patients, 3 melanoma + 3 NSCLC), tested for immunogenicity via HLA-I multimer staining of patient-matched TILs or PBMCs.

**Columns:**

| Column | Description |
|--------|-------------|
| PMHC | Peptide-MHC identifier (format: `allele_peptide`) |
| PATIENT_ID | TESLA patient identifier |
| TISSUE_TYPE | Source of T cells used for validation (PBMC or TIL) |
| MHC | HLA class I allele (e.g. `A*02:01`) |
| ALT_EPI_SEQ | Mutant peptide sequence |
| PEP_LEN | Peptide length (8-14 amino acids) |
| MEASURED_BINDING_AFFINITY | Experimental IC50 from competition binding assay (nM). NA if not measured. |
| NETMHC_PAN_BINDING_AFFINITY | Predicted IC50 from NetMHCpan (nM) |
| TUMOR_ABUNDANCE | Gene expression level. NA if not available. |
| BINDING_STABILITY | Predicted peptide-MHC complex stability (half-life in hours) |
| FRAC_HYDROPHOBIC | Fraction of hydrophobic residues in the peptide |
| AGRETOPICITY | Ratio of mutant to wildtype binding affinity (higher means mutant binds relatively better) |
| FOREIGNNESS | Dissimilarity score comparing mutant peptide to the self-proteome |
| MUTATION_POSITION | 1-based position of the somatic mutation within the peptide |
| NUMBER_PREDICTING | Number of TESLA consortium teams that included this peptide in their predictions |
| VALIDATED | Immunogenicity label. True if the peptide elicited a T cell response in multimer staining. |
| TCR_FLOW_I | T cell response detected by flow cytometry (round I) |
| TCR_FLOW_I_QUANT | Quantitative flow cytometry measurement (round I) |
| TCR_NANOPARTICLE | T cell response detected by nanoparticle assay |
| TCR_FLOW_II | T cell response detected by flow cytometry (round II) |
| TCR_FLOW_II_QUANT | Quantitative flow cytometry measurement (round II) |

**Key statistics:**
- 608 peptides total
- 37 immunogenic (VALIDATED = True)
- 571 non-immunogenic (VALIDATED = False)
- 6 patients (IDs: 1, 2, 3, 10, 12, 16)
- 13 HLA alleles represented

### `s7_pMHC_validation_cohort.xlsx`

Supplementary Table S7 from Wells et al. (2020). Contains 310 peptide-MHC pairs from the second TESLA batch (3 additional melanoma patients), used as an independent validation cohort.

Same column structure as S4 but without the MHC, FRAC_HYDROPHOBIC, NUMBER_PREDICTING, TCR_NANOPARTICLE, and TCR_FLOW_I columns.

**Key statistics:**
- 310 peptides total
- 4 immunogenic (VALIDATED = 1)
- 306 non-immunogenic (VALIDATED = 0)
- 3 patients (IDs: 4, 8, 9)
- 7 HLA alleles represented

## Intended use

This dataset serves as a benchmark for evaluating neoantigen ranking and prioritisation algorithms. The standard evaluation protocol (as used by TESLA consortium teams) is:

1. For each patient, rank all tested peptides by a predicted score.
2. Count how many of the immunogenic peptides (VALIDATED = True) appear in the top 20, top 50, and top 100 predictions.
3. Compute AUPRC (area under precision-recall curve), fraction ranked (FR), and top-twenty immunogenic fraction (TTIF).

The peptide sequences and HLA alleles can be passed directly to binding/presentation prediction tools (MHCflurry, NetMHCpan, MixMHCpred, PRIME) without needing raw sequencing data. The precomputed features (AGRETOPICITY, FOREIGNNESS, BINDING_STABILITY) can be used as additional scoring features or as a comparison baseline.

Note that the wildtype peptide sequences are not included in this table. The AGRETOPICITY and FOREIGNNESS columns provide precomputed values that depend on the wildtype sequence.

## Licensing and attribution

The Wells et al. (2020) paper is published in *Cell* under Elsevier's standard terms. The supplementary data tables are distributed alongside the publication for research use. If you use this data, cite the original paper.

The raw sequencing data underlying the TESLA study (tumor/normal WES, tumor RNA-seq, clinical HLA typing) is available under controlled access on Synapse at [syn21048999](https://www.synapse.org/#!Synapse:syn21048999). Accessing the raw data requires an approved data access request. The supplementary tables used here do not contain individually identifiable genomic data.

## Related resources

- TESLA Synapse portal: https://www.synapse.org/#!Synapse:syn21048999
- Parker Institute for Cancer Immunotherapy: https://www.parkerici.org/
