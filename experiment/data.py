from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    df: pd.DataFrame
    hist_exog: list[str]
    futr_exog: list[str]
    csv_path: str
    unique_id: str


class DatasetBuilder:
    @staticmethod
    def _dedupe_preserve_order(columns: list[str]) -> list[str]:
        return list(dict.fromkeys(columns))

    @classmethod
    def _validate_exog_lists(
        cls,
        hist_exog: list[str],
        futr_exog: list[str],
    ) -> tuple[list[str], list[str]]:
        normalized_hist = cls._dedupe_preserve_order(list(hist_exog))
        normalized_futr = cls._dedupe_preserve_order(list(futr_exog))
        overlap = sorted(set(normalized_hist) & set(normalized_futr))
        if overlap:
            raise ValueError(
                "hist_exog and futr_exog must be disjoint, "
                f"found overlapping columns: {overlap}"
            )
        return normalized_hist, normalized_futr

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = (
            normalized.columns.str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("\u00b0", "deg", regex=False)
            .str.replace("\u63b3", "deg", regex=False)
            .str.replace("/", "_", regex=False)
            .str.replace("\u00b2", "_square", regex=False)
        )
        return normalized

    @staticmethod
    def build_neuralforecast_df(
        df: pd.DataFrame,
        unique_id: str,
        futr_exog: Optional[list[str]] = None,
        hist_exog: Optional[list[str]] = None,
        date_col: str = "date",
        target_col: str = "OT",
        verbose: bool = True,
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        normalized = DatasetBuilder.normalize_columns(df)
        normalized["ds"] = pd.to_datetime(normalized[date_col])
        normalized["y"] = normalized[target_col]
        normalized["unique_id"] = unique_id

        requested_hist, requested_futr = DatasetBuilder._validate_exog_lists(
            hist_exog=hist_exog or [],
            futr_exog=futr_exog or [],
        )
        existing_hist_exog = [
            column for column in requested_hist if column in normalized.columns
        ]
        missing_hist_exog = [
            column for column in requested_hist if column not in normalized.columns
        ]
        existing_futr_exog = [
            column for column in requested_futr if column in normalized.columns
        ]
        missing_futr_exog = [
            column for column in requested_futr if column not in normalized.columns
        ]

        if verbose and missing_hist_exog:
            print("[WARN] Missing historical exogenous columns:")
            for column in missing_hist_exog:
                print("  -", column)

        if verbose and missing_futr_exog:
            print("[WARN] Missing future exogenous columns:")
            for column in missing_futr_exog:
                print("  -", column)

        keep_cols = DatasetBuilder._dedupe_preserve_order(
            ["unique_id", "ds", "y"] + existing_hist_exog + existing_futr_exog
        )
        df_nf = normalized[keep_cols].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        if dropna:
            df_nf = df_nf.dropna().reset_index(drop=True)

        return df_nf, existing_hist_exog, existing_futr_exog

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        unique_id: str,
        date_col: str = "date",
        target_col: str = "OT",
        futr_exog: Optional[list[str]] = None,
        hist_exog: Optional[list[str]] = None,
        verbose: bool = True,
        dropna: bool = True,
    ) -> PreparedDataset:
        raw_df = pd.read_csv(csv_path)
        normalized_df = cls.normalize_columns(raw_df)

        if hist_exog is None and futr_exog is None:
            inferred_hist_exog: list[str] = []
            inferred_futr_exog = [
                column
                for column in normalized_df.columns
                if column not in {date_col, target_col}
            ]
        else:
            inferred_hist_exog = list(hist_exog or [])
            inferred_futr_exog = list(futr_exog or [])

        df_nf, existing_hist_exog, existing_futr_exog = cls.build_neuralforecast_df(
            df=raw_df,
            unique_id=unique_id,
            futr_exog=inferred_futr_exog,
            hist_exog=inferred_hist_exog,
            date_col=date_col,
            target_col=target_col,
            verbose=verbose,
            dropna=dropna,
        )

        return PreparedDataset(
            df=df_nf,
            hist_exog=existing_hist_exog,
            futr_exog=existing_futr_exog,
            csv_path=csv_path,
            unique_id=unique_id,
        )
