from __future__ import annotations

import unittest
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from experiment import RollingTestAnalyzer


class RollingTestAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.history = np.array([10.0, 13.0, 16.0, 20.0, 18.0])
        self.seasonality = 2
        self.eps = 1e-8
        self.mase_scale = 5.0

        self.base_error = np.array(
            [
                [2.0, -2.0, 3.0],
                [-1.0, 2.0, 4.0],
            ]
        )
        self.other_error = np.array(
            [
                [1.0, -3.0, 3.0],
                [-1.0, 3.0, 4.0],
            ]
        )
        self.y_true = np.array(
            [
                [10.0, 20.0, 30.0],
                [8.0, 4.0, 16.0],
            ]
        )
        self.origin_dates = ["2024-01-01", "2024-01-02"]
        self.horizons = [1, 2, 3]
        self.base_raw = self._build_raw(self.base_error, self.y_true)
        self.other_raw = self._build_raw(self.other_error, self.y_true)

        self.summary_error = np.repeat(np.arange(1.0, 12.0).reshape(-1, 1), 2, axis=1)
        self.summary_baseline_error = self.summary_error + 1.0
        self.summary_baseline_error[0] = self.summary_error[0]
        self.summary_y_true = np.full_like(self.summary_error, 10.0)
        self.summary_origin_dates = [f"2024-03-{day:02d}" for day in range(1, 12)]
        self.summary_horizons = [1, 2]
        self.summary_raw = self._build_raw(
            self.summary_error,
            self.summary_y_true,
            origin_dates=self.summary_origin_dates,
            horizons=self.summary_horizons,
        )
        self.summary_baseline_raw = self._build_raw(
            self.summary_baseline_error,
            self.summary_y_true,
            origin_dates=self.summary_origin_dates,
            horizons=self.summary_horizons,
        )

    def _build_raw(
        self,
        error: np.ndarray,
        y_true: np.ndarray,
        origin_dates: list[str] | None = None,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        origin_dates = origin_dates or self.origin_dates
        horizons = horizons or self.horizons

        records: list[dict[str, object]] = []
        for origin_idx, origin_date in enumerate(origin_dates, start=1):
            origin_ts = pd.Timestamp(origin_date)
            for horizon_idx, horizon in enumerate(horizons):
                true_value = float(y_true[origin_idx - 1, horizon_idx])
                error_value = float(error[origin_idx - 1, horizon_idx])
                records.append(
                    {
                        "origin_id": origin_idx,
                        "origin_date": origin_date,
                        "horizon": int(horizon),
                        "target_date": (origin_ts + pd.Timedelta(days=int(horizon))).strftime(
                            "%Y-%m-%d"
                        ),
                        "y_true": true_value,
                        "y_pred": true_value + error_value,
                        "error": error_value,
                    }
                )

        return pd.DataFrame(records).sample(frac=1.0, random_state=7).reset_index(drop=True)

    def _build_analyzer(self, raw: pd.DataFrame | str | Path) -> RollingTestAnalyzer:
        return RollingTestAnalyzer(
            rolling_raw=raw,
            history_actuals=self.history,
            seasonality=self.seasonality,
            eps=self.eps,
        )

    def _manual_overall_metrics(self, error: np.ndarray, y_true: np.ndarray) -> pd.Series:
        ape = np.abs(error) / np.maximum(np.abs(y_true), self.eps)
        metrics = pd.Series(
            {
                "MASE": np.mean(np.abs(error)) / self.mase_scale,
                "RMSE": np.sqrt(np.mean(np.square(error))),
                "WAPE(%)": np.sum(np.abs(error)) / np.maximum(np.sum(np.abs(y_true)), self.eps) * 100,
                "Bias": np.mean(error),
                "MAPE(%)": np.mean(ape) * 100,
            },
            dtype=float,
            name="overall",
        )
        return metrics.loc[list(RollingTestAnalyzer.OVERALL_METRIC_COLUMNS)]

    def _manual_horizon_loss_matrix(
        self,
        error: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        losses = pd.DataFrame(
            {
                "MAE": np.mean(np.abs(error), axis=0),
                "RMSE": np.sqrt(np.mean(np.square(error), axis=0)),
                "Bias": np.mean(error, axis=0),
            },
            index=index,
            dtype=float,
        )
        return losses.loc[:, list(RollingTestAnalyzer.HORIZON_METRIC_COLUMNS)]

    def _manual_window_loss_matrix(
        self,
        error: np.ndarray,
        y_true: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        ape = np.abs(error) / np.maximum(np.abs(y_true), self.eps)
        losses = pd.DataFrame(
            {
                "MAE": np.mean(np.abs(error), axis=1),
                "RMSE": np.sqrt(np.mean(np.square(error), axis=1)),
                "MAPE(%)": np.mean(ape, axis=1) * 100,
            },
            index=index,
            dtype=float,
        )
        return losses.loc[:, list(RollingTestAnalyzer.WINDOW_METRIC_COLUMNS)]

    def _manual_window_summary(
        self,
        losses: pd.DataFrame,
        baseline_losses: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        summary = pd.DataFrame(
            {
                "mean": losses.mean(axis=0),
                "median": losses.median(axis=0),
                "std": losses.std(axis=0, ddof=0),
                "worst10%": losses.apply(
                    lambda values: values.nlargest(max(1, ceil(len(values) * 0.1))).mean(),
                    axis=0,
                ),
                "max": losses.max(axis=0),
            },
            dtype=float,
        )
        summary = summary.loc[
            list(RollingTestAnalyzer.WINDOW_METRIC_COLUMNS),
            list(RollingTestAnalyzer.WINDOW_SUMMARY_COLUMNS),
        ]

        if baseline_losses is not None:
            win_rate = pd.Series(
                {
                    metric: float((losses[metric] < baseline_losses[metric]).mean())
                    for metric in RollingTestAnalyzer.WINDOW_METRIC_COLUMNS
                },
                dtype=float,
            )
            summary["win_rate"] = win_rate.loc[list(RollingTestAnalyzer.WINDOW_METRIC_COLUMNS)]

        return summary

    def test_builds_sorted_error_and_value_matrices(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)

        np.testing.assert_allclose(analyzer.error, self.base_error)
        np.testing.assert_allclose(analyzer.y_true, self.y_true)
        np.testing.assert_allclose(analyzer.y_pred, self.y_true + self.base_error)
        self.assertEqual(analyzer.origin_dates.tolist(), self.origin_dates)
        self.assertEqual(analyzer.horizons.tolist(), self.horizons)

    def test_supports_csv_path_input(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rolling_test_raw.csv"
            self.base_raw.to_csv(csv_path, index=False)
            analyzer = self._build_analyzer(csv_path)

        np.testing.assert_allclose(analyzer.error, self.base_error)

    def test_overall_metrics_follow_expected_formulas(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)
        expected = self._manual_overall_metrics(self.base_error, self.y_true)

        assert_series_equal(analyzer.overall_metrics(), expected, check_exact=False, atol=1e-12)

    def test_loss_matrix_returns_scope_specific_views(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)
        expected_horizon = self._manual_horizon_loss_matrix(
            self.base_error,
            index=pd.Index(self.horizons, name="horizon"),
        )
        expected_window = self._manual_window_loss_matrix(
            self.base_error,
            self.y_true,
            index=pd.Index(self.origin_dates, name="origin_date"),
        )

        actual_horizon = analyzer.loss_matrix("horizon")
        actual_window = analyzer.loss_matrix("window")

        self.assertEqual(actual_horizon.columns.tolist(), list(RollingTestAnalyzer.HORIZON_METRIC_COLUMNS))
        self.assertEqual(actual_window.columns.tolist(), list(RollingTestAnalyzer.WINDOW_METRIC_COLUMNS))

        assert_frame_equal(actual_horizon, expected_horizon, check_exact=False, atol=1e-12)
        assert_frame_equal(actual_window, expected_window, check_exact=False, atol=1e-12)

    def test_window_loss_summary_returns_expected_statistics(self) -> None:
        analyzer = self._build_analyzer(self.summary_raw)
        window_losses = self._manual_window_loss_matrix(
            self.summary_error,
            self.summary_y_true,
            index=pd.Index(self.summary_origin_dates, name="origin_date"),
        )
        expected = self._manual_window_summary(window_losses)

        assert_frame_equal(
            analyzer.loss_summary("window"),
            expected,
            check_exact=False,
            atol=1e-12,
        )

    def test_window_loss_summary_with_baseline_adds_win_rate(self) -> None:
        analyzer = self._build_analyzer(self.summary_raw)
        baseline = self._build_analyzer(self.summary_baseline_raw)
        window_losses = self._manual_window_loss_matrix(
            self.summary_error,
            self.summary_y_true,
            index=pd.Index(self.summary_origin_dates, name="origin_date"),
        )
        baseline_losses = self._manual_window_loss_matrix(
            self.summary_baseline_error,
            self.summary_y_true,
            index=pd.Index(self.summary_origin_dates, name="origin_date"),
        )
        expected = self._manual_window_summary(window_losses, baseline_losses)
        actual = analyzer.loss_summary("window", baseline=baseline)

        assert_frame_equal(actual, expected, check_exact=False, atol=1e-12)
        self.assertAlmostEqual(actual.loc["MAE", "win_rate"], 10.0 / 11.0, places=12)
        self.assertAlmostEqual(actual.loc["RMSE", "win_rate"], 10.0 / 11.0, places=12)
        self.assertAlmostEqual(actual.loc["MAPE(%)", "win_rate"], 10.0 / 11.0, places=12)

    def test_loss_summary_rejects_horizon_scope(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)

        with self.assertRaisesRegex(ValueError, "Only window-wise loss summary is supported"):
            analyzer.loss_summary("horizon")

    def test_window_loss_summary_requires_matching_indexes(self) -> None:
        analyzer = self._build_analyzer(self.summary_raw)
        shifted_baseline_raw = self._build_raw(
            self.summary_baseline_error,
            self.summary_y_true,
            origin_dates=[*self.summary_origin_dates[:-1], "2024-04-01"],
            horizons=self.summary_horizons,
        )
        shifted_baseline = self._build_analyzer(shifted_baseline_raw)

        with self.assertRaisesRegex(ValueError, "different indexes"):
            analyzer.loss_summary("window", baseline=shifted_baseline)

    def test_zero_targets_use_eps_safely(self) -> None:
        zero_error = np.array([[1.0, -1.0]])
        zero_true = np.array([[0.0, 0.0]])
        zero_raw = pd.DataFrame(
            [
                {
                    "origin_id": 1,
                    "origin_date": "2024-02-01",
                    "horizon": 2,
                    "target_date": "2024-02-03",
                    "y_true": 0.0,
                    "y_pred": -1.0,
                    "error": -1.0,
                },
                {
                    "origin_id": 1,
                    "origin_date": "2024-02-01",
                    "horizon": 1,
                    "target_date": "2024-02-02",
                    "y_true": 0.0,
                    "y_pred": 1.0,
                    "error": 1.0,
                },
            ]
        )
        analyzer = RollingTestAnalyzer(
            rolling_raw=zero_raw,
            history_actuals=[1.0, 2.0, 3.0],
            seasonality=1,
            eps=self.eps,
        )

        overall = analyzer.overall_metrics()
        self.assertTrue(np.isfinite(overall["MAPE(%)"]))
        self.assertTrue(np.isfinite(overall["WAPE(%)"]))
        self.assertAlmostEqual(overall["MAPE(%)"], (1.0 / self.eps) * 100, places=2)
        self.assertAlmostEqual(overall["WAPE(%)"], (2.0 / self.eps) * 100, places=2)
        np.testing.assert_allclose(analyzer.error, zero_error)
        np.testing.assert_allclose(analyzer.y_true, zero_true)

    def test_validation_errors_cover_common_bad_inputs(self) -> None:
        missing_error = self.base_raw.drop(columns=["error"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self._build_analyzer(missing_error)

        inconsistent_error = self.base_raw.copy()
        inconsistent_error.loc[inconsistent_error.index[0], "error"] += 1.0
        with self.assertRaisesRegex(ValueError, "error column must match"):
            self._build_analyzer(inconsistent_error)

        inconsistent_horizon = self.base_raw.copy()
        drop_index = inconsistent_horizon[
            (inconsistent_horizon["origin_id"] == 2) & (inconsistent_horizon["horizon"] == 3)
        ].index[0]
        inconsistent_horizon = inconsistent_horizon.drop(index=drop_index)
        with self.assertRaisesRegex(ValueError, "full shared horizon set"):
            self._build_analyzer(inconsistent_horizon)

        with self.assertRaisesRegex(ValueError, "longer than seasonality"):
            RollingTestAnalyzer(
                rolling_raw=self.base_raw,
                history_actuals=[1.0, 2.0],
                seasonality=2,
                eps=self.eps,
            )


if __name__ == "__main__":
    unittest.main()

