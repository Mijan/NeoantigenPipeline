"""Custom exceptions for the neoantigen prediction pipeline."""

from __future__ import annotations


class NeoantigenPipelineError(Exception):
    """Base exception for all neoantigen pipeline errors.

    All pipeline-specific exceptions inherit from this class, allowing
    callers to catch any pipeline error with a single except clause.
    """


class VCFParsingError(NeoantigenPipelineError):
    """Raised when VCF file parsing fails or produces unexpected results.

    Examples include malformed records, missing required INFO fields,
    or CSQ annotation format mismatches.
    """


class HLAParsingError(NeoantigenPipelineError):
    """Raised when HLA type file parsing fails.

    Covers OptiType TSV format errors and invalid allele notation.
    """


class ProteomeError(NeoantigenPipelineError):
    """Raised when reference proteome operations fail.

    Includes FASTA parsing errors and missing transcript lookups.
    """


class PredictionError(NeoantigenPipelineError):
    """Raised when MHC binding prediction fails.

    Covers model loading failures and prediction runtime errors.
    """


class ConfigurationError(NeoantigenPipelineError):
    """Raised when pipeline configuration is invalid or missing required fields.

    Includes YAML parsing errors and invalid parameter values.
    """
