"""Utilities for filtering training molecules that overlap with a validation set
based on structural identifiers (DrugID / InChIKey / canonical SMILES).

These helpers are kept framework-agnostic so they can be reused from notebooks
or scripts before creating the final PyTorch datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

try:  # RDKit is optional at runtime (not available in the CLI harness)
    from rdkit import Chem
    from rdkit.Chem import inchi as rd_inchi
except Exception:  # pragma: no cover - runtime falls back when RDKit missing
    Chem = None
    rd_inchi = None


@dataclass(frozen=True)
class OverlapReport:
    """Metadata describing which identifiers caused molecules to be dropped."""

    by_id: set[str]
    by_inchikey: set[str]
    by_smiles: set[str]

    @property
    def union(self) -> set[str]:
        return set(self.by_id) | set(self.by_inchikey) | set(self.by_smiles)

    def as_dict(self) -> dict[str, Sequence[str]]:
        return {
            "by_id": sorted(self.by_id),
            "by_inchikey": sorted(self.by_inchikey),
            "by_smiles": sorted(self.by_smiles),
            "total": sorted(self.union),
        }


def _canonical_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _inchikey_from_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles or rd_inchi is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles) if Chem else None
    if mol is None:
        return None
    try:
        return rd_inchi.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - rdkit raises on unsupported mols
        return None


def prepare_structure_table(
    df: pd.DataFrame,
    *,
    id_col: str,
    smiles_col: Optional[str] = None,
    inchikey_col: Optional[str] = None,
    name: str = "train",
) -> pd.DataFrame:
    """Return dataframe with columns DrugID / SMILES / InChIKey / key.

    The caller can supply whichever identifier columns are available. When
    InChIKey is not present we fall back to RDKit's canonical SMILES as
    the structural key. The resulting table can be passed to
    :func:`find_structure_overlap`.
    """

    required = {id_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns {sorted(missing)} in {name} table")

    struct = df.copy()
    struct.rename(columns={id_col: "DrugID"}, inplace=True)

    if smiles_col and smiles_col in struct.columns:
        struct.rename(columns={smiles_col: "SMILES"}, inplace=True)
    else:
        struct["SMILES"] = None

    if inchikey_col and inchikey_col in struct.columns:
        struct.rename(columns={inchikey_col: "InChIKey"}, inplace=True)
    else:
        struct["InChIKey"] = None

    struct["SMILES"] = struct["SMILES"].astype(str).where(struct["SMILES"].notna())
    struct["InChIKey"] = struct["InChIKey"].astype(str).where(struct["InChIKey"].notna())

    struct["can_smiles"] = struct["SMILES"].map(_canonical_smiles)
    # Prefer provided InChIKey, otherwise attempt to derive from SMILES
    struct["key"] = struct["InChIKey"].where(
        struct["InChIKey"].notna() & struct["InChIKey"].str.len() > 0,
        struct["can_smiles"],
    )

    # Fill missing keys with RDKit generated InChIKey when possible
    needs_key = struct["key"].isna() | (struct["key"].astype(str).str.len() == 0)
    if needs_key.any():
        derived = struct.loc[needs_key, "SMILES"].map(_inchikey_from_smiles)
        struct.loc[needs_key, "key"] = derived.where(derived.notna(), struct.loc[needs_key, "key"])

    struct.drop_duplicates(subset=["DrugID"], keep="first", inplace=True)
    return struct


def find_structure_overlap(
    train_struct: pd.DataFrame,
    val_struct: pd.DataFrame,
) -> OverlapReport:
    """Return overlapping training DrugIDs based on ID / InChIKey / SMILES."""

    val_ids = set(val_struct["DrugID"].dropna().astype(str))
    train_ids = set(train_struct["DrugID"].dropna().astype(str))

    by_id = train_ids & val_ids

    val_inchis = set(
        val_struct["InChIKey"].dropna().astype(str)
    )
    val_keys = set(val_struct["key"].dropna().astype(str))
    val_smiles = set(val_struct["can_smiles"].dropna().astype(str))

    train_inchis = train_struct.set_index("DrugID")["InChIKey"].dropna().astype(str)
    train_keys = train_struct.set_index("DrugID")["key"].dropna().astype(str)
    train_smiles = train_struct.set_index("DrugID")["can_smiles"].dropna().astype(str)

    by_inchikey = {
        drug_id
        for drug_id, inchikey in train_inchis.items()
        if inchikey in val_inchis
    }

    # Structural keys fallback to canonical smiles when InChIKey absent
    by_structure_key = {
        drug_id
        for drug_id, key in train_keys.items()
        if key and key in val_keys
    }

    by_smiles = {
        drug_id
        for drug_id, smiles in train_smiles.items()
        if smiles and smiles in val_smiles
    }

    # Merge structure overlaps (includes InChIKey hits)
    by_inchikey |= by_structure_key

    return OverlapReport(by_id=by_id, by_inchikey=by_inchikey, by_smiles=by_smiles)


def drop_overlapping_entries(
    df: pd.DataFrame,
    drop_ids: Iterable[str],
    *,
    id_col: str = "DrugID",
) -> pd.DataFrame:
    """Return dataframe without rows whose identifier matches ``drop_ids``."""

    drop_ids = {str(i) for i in drop_ids}
    if not drop_ids:
        return df
    mask = ~df[id_col].astype(str).isin(drop_ids)
    return df.loc[mask].copy()


def filter_training_by_validation(
    df_scores: pd.DataFrame,
    df_labels: pd.DataFrame,
    train_struct: pd.DataFrame,
    val_struct: pd.DataFrame,
    *,
    id_col: str = "DrugID",
) -> tuple[pd.DataFrame, pd.DataFrame, OverlapReport]:
    """Remove validation-overlapping molecules from the training matrices.

    Parameters
    ----------
    df_scores:
        DataFrame whose columns correspond to the training compound identifiers.
        Columns whose ID appears in ``drop_ids`` will be removed.
    df_labels:
        Training label table containing a ``DrugID`` column.
    train_struct / val_struct:
        Structure tables prepared with :func:`prepare_structure_table`.
    id_col:
        Column name that identifies compounds inside ``df_labels``.

    Returns
    -------
    filtered_scores, filtered_labels, overlap_report
    """

    overlap = find_structure_overlap(train_struct, val_struct)
    to_drop = overlap.union
    if not to_drop:
        return df_scores, df_labels, overlap

    drop_cols = [c for c in df_scores.columns if c in to_drop]
    filtered_scores = df_scores.drop(columns=drop_cols, errors="ignore")
    filtered_labels = drop_overlapping_entries(df_labels, to_drop, id_col=id_col)

    return filtered_scores, filtered_labels, overlap


__all__ = [
    "OverlapReport",
    "prepare_structure_table",
    "find_structure_overlap",
    "drop_overlapping_entries",
    "filter_training_by_validation",
]
