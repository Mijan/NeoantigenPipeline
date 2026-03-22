# HCC1395 Breast Cancer Cell Line Dataset

## Source

This data was downloaded from the pVACtools Introduction Course, maintained by the Griffith Lab at Washington University School of Medicine in St. Louis.

**Download URL:**
```
wget https://raw.githubusercontent.com/griffithlab/pVACtools_Intro_Course/main/HCC1395_inputs.zip
```

**Course materials:** https://course.pvactools.org/

The data originates from the **HCC1395** breast cancer cell line (triple-negative, ductal carcinoma) and its matched normal lymphoblastoid cell line **HCC1395BL**. This cell line pair is widely used as a reference standard in cancer bioinformatics for benchmarking somatic variant calling, neoantigen prediction, and related workflows.

## Files

| File | Description |
|------|-------------|
| `annotated.expression.vcf.gz` | VEP-annotated somatic VCF with expression and coverage information. Contains missense, indel, and frameshift variants identified from tumor-normal comparison. |
| `annotated.expression.vcf.gz.tbi` | Tabix index for the VCF. |
| `phased.vcf.gz` | Phased tumor-germline VCF for proximal variant correction. Contains somatic and germline variants with phase information to identify in-phase variants that may alter predicted peptide sequences. |
| `phased.vcf.gz.tbi` | Tabix index for the phased VCF. |
| `optitype_normal_result.tsv` | HLA class I typing results from OptiType, run on the matched normal sample. |
| `Homo_sapiens.GRCh38.pep.all.fa.gz` | Ensembl GRCh38 reference proteome (all protein sequences). Used for self-similarity filtering of neoantigen candidates. |
| `star-fusion.fusion_predictions.tsv` | Gene fusion predictions from STAR-Fusion. |
| `agfusion_results/` | AGFusion output for annotating gene fusions with protein domain and sequence information. |
| `HCC1395.splice_junctions.tsv` | Splice junction data for splice-site-derived neoantigen prediction. |

## HLA Types

Determined by OptiType (class I) and clinical typing (class II):

**Class I:**
- HLA-A\*29:02
- HLA-B\*45:01
- HLA-B\*82:02 (note: this is homozygous for HLA-A, only one A allele detected)
- HLA-C\*06:02

**Class II** (from pVACtools course documentation, not in the OptiType file):
- DQA1\*03:03
- DQB1\*03:02
- DRB1\*04:05

## VCF annotations

The somatic VCF has been preprocessed with the following tools and annotations:

- **Variant calling**: somatic variants identified from tumor-normal paired analysis.
- **VEP (Variant Effect Predictor)**: functional consequence annotation including gene, transcript, protein change (HGVSp), and consequence type. VEP was run with the Wildtype and Downstream plugins required by pVACtools.
- **Expression**: gene-level expression values (from RNA-seq) annotated into the VCF INFO or FORMAT fields.
- **Coverage**: read depth and variant allele frequency information for both tumor DNA and RNA.
- **Reference genome**: GRCh38 (hg38).

## Intended use

This dataset is used for developing and testing the neoantigen prediction pipeline end-to-end. It provides a complete set of inputs for running from VCF through peptide generation, MHC binding prediction, and candidate ranking.

**Important:** this dataset does NOT contain experimental immunogenicity validation data. There are no ground-truth labels indicating which predicted neoantigens actually elicit T cell responses. It is suitable for pipeline development, debugging, and concordance analysis (comparing output with pVACtools results), but not for benchmarking ranking accuracy. Use the TESLA dataset for benchmarking.

## Licensing and attribution

pVACtools is open-source software developed by the Griffith Lab and distributed under the BSD 3-Clause License.

> Hundal, J., Kiwala, S., McMichael, J. et al.
> "pVACtools: A Computational Toolkit to Identify and Visualize Cancer Neoantigens."
> *Cancer Immunology Research* 8(3), 409-420 (2020).
> DOI: [10.1158/2326-6066.CIR-19-0401](https://doi.org/10.1158/2326-6066.CIR-19-0401)

The HCC1395 cell line data used in the pVACtools course is derived from publicly available resources. The original cell line characterization and sequencing data are described in:

> Griffith Lab Precision Medicine Bioinformatics Course:
> https://pmbio.org/

Additional context on the HCC1395 cell line as a reference standard:

> Fang, L.T., Zhu, B., Zhao, Y. et al.
> "Establishing community reference samples, data and call sets for benchmarking cancer mutation detection using whole-genome sequencing."
> *Nature Biotechnology* 39, 1151-1160 (2021).
> DOI: [10.1038/s41587-021-00993-6](https://doi.org/10.1038/s41587-021-00993-6)

## Related resources

- pVACtools documentation: https://pvactools.readthedocs.io/
- pVACtools GitHub: https://github.com/griffithlab/pVACtools
- pVACtools course: https://course.pvactools.org/
- pVACview online server: https://pvacview.org/
