# NeoantigenPipeline

## Purpose

A pipeline for **neoantigen prediction** that processes somatic variant data (VCF files and similar formats) through to peptide-MHC binding candidate predictions.

The pipeline is intended to cover the following stages:

1. **Variant ingestion** — parse VCF/MAF files containing somatic mutations
2. **Peptide generation** — derive mutant peptide sequences around each variant
3. **MHC binding prediction** — score peptide-MHC affinity for candidate neoantigens
4. **Filtering & ranking** — prioritise candidates by binding strength and other criteria

------------------------------------------------------------------------

## Repository Structure

    .
    ├── .github/workflows/ci-quality.yml
    ├── pyproject.toml
    ├── requirements.txt
    ├── src/neoantigen_pipeline/
    │   ├── __init__.py
    │   ├── pipeline.py
    │   └── profiler.py
    ├── tests/
    ├── tox.ini
    └── README.md

------------------------------------------------------------------------

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

------------------------------------------------------------------------

## Quality Checks

```bash
tox              # run all environments
tox -e lint      # ruff linting
tox -e type      # mypy type checking
tox -e format    # auto-format
```

------------------------------------------------------------------------

## Training Profiler

`src/neoantigen_pipeline/profiler.py` provides lightweight per-phase wall-clock profiling with optional CUDA synchronisation and W&B integration. See the class docstrings for usage.

| Class | Role |
|---|---|
| `PhaseTimer` | Times named phases within each batch; optionally syncs CUDA |
| `ProfileLogger` | Writes per-batch and per-epoch CSV logs to disk |
| `WandbLogger` | Logs epoch metrics and phase timings to Weights & Biases |

------------------------------------------------------------------------

## Optional predictors

### HLApollo predictor

HLApollo is a transformer-based MHC-I presentation predictor from Genentech.
It is distributed as a compiled Linux binary and requires separate installation.

**System dependencies:**
- Linux (x86_64)
- libatlas or openblas (`sudo pacman -S openblas` on Arch, `sudo apt-get install libatlas-base-dev` on Debian/Ubuntu)
- git-lfs (`sudo pacman -S git-lfs` / `sudo apt-get install git-lfs`)

**Installation:**
```bash
git lfs install
mkdir -p tools
cd tools
git clone https://github.com/Genentech/HLApollo.git
chmod +x HLApollo/HLA-Apollo
cd ..
```

**Verify:**
```bash
./tools/HLApollo/HLA-Apollo tools/HLApollo/example.csv /tmp/hlapollo_test.csv
```

**Docker alternative (non-Linux or if binary fails):**
```bash
cd tools/HLApollo
docker build -t hla-apollo .
cd ../..
```

Then set `hlapollo.docker_image: "hla-apollo"` in your config.

**Enable in config:**
Set `hlapollo.enabled: true` in `configs/default.yaml` or pass programmatically.

------------------------------------------------------------------------

### ESM-2 protein embeddings (optional)

ESM-2 protein language model embeddings provide structural context features
for improved neoantigen ranking.

**Installation:**
```bash
pip install 'neoantigen-pipeline[esm]'
```

This installs `fair-esm`, `torch`, and `h5py`. A CUDA-capable GPU is recommended
but not required (CPU inference works but is slower).

**Enable in config:**
Set `esm.enabled: true` in `configs/default.yaml`.
Embeddings are cached to `results/esm_cache.h5` after first computation.

------------------------------------------------------------------------

## License

MIT License © 2026 Jan Mikelson
