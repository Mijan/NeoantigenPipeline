"""Reference proteome FASTA loading and sequence lookup.

Loads Ensembl peptide FASTA files and provides transcript-based protein
sequence retrieval for wildtype peptide generation.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import IO

from neoantigen_pipeline.exceptions import ProteomeError


class ProteomeDB:
    """In-memory reference proteome database backed by an Ensembl FASTA file.

    Ensembl peptide FASTA headers have the format::

        >ENSP00000... pep ... transcript:ENST00000... ...

    The database indexes sequences by their Ensembl transcript ID so they
    can be looked up efficiently during peptide generation.

    Args:
        fasta_path: Path to the (optionally gzip-compressed) Ensembl
            peptide FASTA file.

    Raises:
        ProteomeError: If the file cannot be read or contains no sequences.
    """

    # Regex to extract transcript ID from Ensembl FASTA header
    _TRANSCRIPT_RE = re.compile(r"transcript:(\S+)")
    # Also try gene/protein ID patterns for resilience
    _PROTEIN_RE = re.compile(r"^>(\S+)")

    def __init__(self, fasta_path: str) -> None:
        self._fasta_path = fasta_path
        self._logger = logging.getLogger(type(self).__qualname__)
        self._sequences: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Parse the FASTA file and populate the internal sequence index.

        Raises:
            ProteomeError: If the file cannot be opened or is empty.
        """
        path = Path(self._fasta_path)
        self._logger.info("Loading proteome from '%s'", self._fasta_path)

        try:
            if path.suffix == ".gz" or path.name.endswith(".fa.gz"):
                opener = gzip.open(path, "rt", encoding="utf-8")
            else:
                opener = open(path, encoding="utf-8")  # noqa: SIM115

            with opener as fh:
                self._parse_fasta(fh)
        except OSError as exc:
            raise ProteomeError(
                f"Cannot open proteome file '{self._fasta_path}': {exc}"
            ) from exc

        if not self._sequences:
            raise ProteomeError(
                f"Proteome file '{self._fasta_path}' contained no parseable sequences"
            )

        self._logger.info("Loaded %d protein sequences", len(self._sequences))

    def _parse_fasta(self, fh: IO[str]) -> None:
        """Iterate over a FASTA file handle and populate self._sequences.

        Both transcript ID and protein ID are used as keys to maximise
        lookup flexibility.

        Args:
            fh: An open text file handle.
        """
        current_keys: list[str] = []
        seq_parts: list[str] = []

        def _commit() -> None:
            if current_keys and seq_parts:
                seq = "".join(seq_parts)
                for key in current_keys:
                    self._sequences[key] = seq

        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _commit()
                current_keys = []
                seq_parts = []
                current_keys.extend(self._parse_header_keys(line))
            else:
                seq_parts.append(line)

        _commit()

    def _parse_header_keys(self, header: str) -> list[str]:
        """Extract all usable keys from a FASTA header line.

        Tries to extract the transcript ID (primary key) and falls back
        to the protein ID.

        Args:
            header: Full FASTA header line starting with ">".

        Returns:
            List of identifier strings to index this sequence under.
        """
        keys: list[str] = []

        # Primary: transcript ID from "transcript:ENST..." token
        transcript_match = self._TRANSCRIPT_RE.search(header)
        if transcript_match:
            transcript_id = transcript_match.group(1)
            # Strip version suffix (e.g. ENST00000123456.3 -> ENST00000123456)
            keys.append(transcript_id)
            base_id = transcript_id.split(".")[0]
            if base_id != transcript_id:
                keys.append(base_id)

        # Secondary: protein ID (first token after ">")
        protein_match = self._PROTEIN_RE.match(header)
        if protein_match:
            protein_id = protein_match.group(1)
            keys.append(protein_id)
            base_id = protein_id.split(".")[0]
            if base_id != protein_id:
                keys.append(base_id)

        return keys

    def get_sequence(self, transcript_id: str) -> str | None:
        """Retrieve a protein sequence by transcript ID.

        Tries both the full versioned ID and the version-stripped base ID.

        Args:
            transcript_id: Ensembl transcript identifier.

        Returns:
            Amino acid sequence string, or None if not found.
        """
        seq = self._sequences.get(transcript_id)
        if seq is not None:
            return seq
        # Try stripping version number
        base_id = transcript_id.split(".")[0]
        return self._sequences.get(base_id)

    def get_protein_sequence_for_transcript(self, transcript_id: str) -> str | None:
        """Retrieve a protein sequence by transcript ID (alias for get_sequence).

        Provided as a semantically explicit alternative to get_sequence.

        Args:
            transcript_id: Ensembl transcript identifier.

        Returns:
            Amino acid sequence string, or None if not found.
        """
        return self.get_sequence(transcript_id)

    @property
    def size(self) -> int:
        """Number of protein sequences indexed.

        Returns:
            Count of entries (note: may exceed number of unique proteins if
            multiple keys point to the same sequence).
        """
        return len(self._sequences)
