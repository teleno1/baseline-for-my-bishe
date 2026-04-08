from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    df: pd.DataFrame
    futr_exog: list[str]
    csv_path: str
    unique_id: str


class DatasetBuilder:
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = (
            normalized.columns.str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("°", "deg", regex=False)
            .str.replace("掳", "deg", regex=False)
            .str.replace("/", "_", regex=False)
            .str.replace("²", "_square", regex=False)
        )
        return normalized

    @staticmethod
    def build_neuralforecast_df(
        df: pd.DataFrame,
        unique_id: str,
        futr_exog: list[str],
        date_col: str = "date",
        target_col: str = "OT",
        verbose: bool = True,
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        normalized = DatasetBuilder.normalize_columns(df)
        normalized["ds"] = pd.to_datetime(normalized[date_col])
        normalized["y"] = normalized[target_col]
        normalized["unique_id"] = unique_id

        existing_exog = [column for column in futr_exog if column in normalized.columns]
        missing_exog = [column for column in futr_exog if column not in normalized.columns]

        if verbose and missing_exog:
            print("[WARN] Missing future exogenous columns:")
            for column in missing_exog:
                print("  -", column)

        keep_cols = ["unique_id", "ds", "y"] + existing_exog
        df_nf = normalized[keep_cols].copy()
        df_nf = df_nf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        if dropna:
            df_nf = df_nf.dropna().reset_index(drop=True)

        return df_nf, existing_exog

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        unique_id: str,
        date_col: str = "date",
        target_col: str = "OT",
        futr_exog: Optional[list[str]] = None,
        verbose: bool = True,
        dropna: bool = True,
    ) -> PreparedDataset:
        raw_df = pd.read_csv(csv_path)
        normalized_df = cls.normalize_columns(raw_df)

        inferred_exog = futr_exog
        if inferred_exog is None:
            inferred_exog = [
                column
                for column in normalized_df.columns
                if column not in {date_col, target_col}
            ]

        df_nf, existing_exog = cls.build_neuralforecast_df(
            df=raw_df,
            unique_id=unique_id,
            futr_exog=inferred_exog,
            date_col=date_col,
            target_col=target_col,
            verbose=verbose,
            dropna=dropna,
        )

        return PreparedDataset(
            df=df_nf,
            futr_exog=existing_exog,
            csv_path=csv_path,
            unique_id=unique_id,
        )
