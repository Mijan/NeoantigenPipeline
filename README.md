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

## License

MIT License © 2026 Jan Mikelson
