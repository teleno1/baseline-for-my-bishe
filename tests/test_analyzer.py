from __future__ import annotations

import unittest
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

    def _manual_loss_matrix(
        self,
        error: np.ndarray,
        y_true: np.ndarray,
        axis: int,
        index: pd.Index,
    ) -> pd.DataFrame:
        ape = np.abs(error) / np.maximum(np.abs(y_true), self.eps)
        losses = pd.DataFrame(
            {
                "MASE": np.mean(np.abs(error), axis=axis) / self.mase_scale,
                "RMSE": np.sqrt(np.mean(np.square(error), axis=axis)),
                "WAPE": (
                    np.sum(np.abs(error), axis=axis)
                    / np.maximum(np.sum(np.abs(y_true), axis=axis), self.eps)
                    * 100
                ),
                "Bias": np.mean(error, axis=axis),
                "MAPE": np.mean(ape, axis=axis) * 100,
            },
            index=index,
            dtype=float,
        )
        return losses.loc[:, RollingTestAnalyzer.METRIC_COLUMNS]

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
        ape = np.abs(self.base_error) / np.abs(self.y_true)
        expected = pd.Series(
            {
                "MASE": np.mean(np.abs(self.base_error)) / self.mase_scale,
                "RMSE": np.sqrt(np.mean(np.square(self.base_error))),
                "WAPE": np.sum(np.abs(self.base_error)) / np.sum(np.abs(self.y_true)) * 100,
                "Bias": np.mean(self.base_error),
                "MAPE": np.mean(ape) * 100,
            },
            dtype=float,
            name="overall",
        )
        expected = expected.loc[list(RollingTestAnalyzer.METRIC_COLUMNS)]

        assert_series_equal(analyzer.overall_metrics(), expected, check_exact=False, atol=1e-12)

    def test_loss_matrix_returns_horizon_and_window_views(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)
        expected_horizon = self._manual_loss_matrix(
            self.base_error,
            self.y_true,
            axis=0,
            index=pd.Index(self.horizons, name="horizon"),
        )
        expected_window = self._manual_loss_matrix(
            self.base_error,
            self.y_true,
            axis=1,
            index=pd.Index(self.origin_dates, name="origin_date"),
        )

        assert_frame_equal(
            analyzer.loss_matrix("horizon"),
            expected_horizon,
            check_exact=False,
            atol=1e-12,
        )
        assert_frame_equal(
            analyzer.loss_matrix("window"),
            expected_window,
            check_exact=False,
            atol=1e-12,
        )

    def test_loss_summary_matches_loss_matrix_aggregation(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)

        for scope in ("horizon", "window"):
            with self.subTest(scope=scope):
                losses = analyzer.loss_matrix(scope)
                expected = losses.agg(["min", "max", "mean", "median"]).T
                expected["std"] = losses.std(axis=0, ddof=0)
                expected = expected.loc[:, ["min", "max", "mean", "median", "std"]]
                assert_frame_equal(
                    analyzer.loss_summary(scope),
                    expected,
                    check_exact=False,
                    atol=1e-12,
                )

    def test_overall_equals_mean_for_mase_bias_mape_only(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)
        overall = analyzer.overall_metrics()
        horizon_mean = analyzer.loss_matrix("horizon").mean(axis=0)
        window_mean = analyzer.loss_matrix("window").mean(axis=0)

        for metric in ("MASE", "Bias", "MAPE"):
            with self.subTest(metric=metric):
                self.assertAlmostEqual(overall[metric], horizon_mean[metric], places=12)
                self.assertAlmostEqual(overall[metric], window_mean[metric], places=12)

        self.assertGreater(abs(overall["RMSE"] - horizon_mean["RMSE"]), 1e-6)
        self.assertGreater(abs(overall["RMSE"] - window_mean["RMSE"]), 1e-6)
        self.assertGreater(abs(overall["WAPE"] - horizon_mean["WAPE"]), 1e-6)
        self.assertGreater(abs(overall["WAPE"] - window_mean["WAPE"]), 1e-6)

    def test_compare_winrate_uses_absolute_bias_and_counts_ties_as_zero(self) -> None:
        analyzer_a = self._build_analyzer(self.base_raw)
        analyzer_b = self._build_analyzer(self.other_raw)

        for scope in ("horizon", "window"):
            with self.subTest(scope=scope):
                left = self._manual_loss_matrix(
                    self.base_error,
                    self.y_true,
                    axis=0 if scope == "horizon" else 1,
                    index=(
                        pd.Index(self.horizons, name="horizon")
                        if scope == "horizon"
                        else pd.Index(self.origin_dates, name="origin_date")
                    ),
                )
                right = self._manual_loss_matrix(
                    self.other_error,
                    self.y_true,
                    axis=0 if scope == "horizon" else 1,
                    index=(
                        pd.Index(self.horizons, name="horizon")
                        if scope == "horizon"
                        else pd.Index(self.origin_dates, name="origin_date")
                    ),
                )

                expected = {}
                for metric in RollingTestAnalyzer.METRIC_COLUMNS:
                    left_values = left[metric]
                    right_values = right[metric]
                    if metric == "Bias":
                        left_values = left_values.abs()
                        right_values = right_values.abs()
                    expected[metric] = float((left_values < right_values).mean())

                expected_series = pd.Series(
                    expected,
                    dtype=float,
                    name=f"{scope}_winrate",
                ).loc[list(RollingTestAnalyzer.METRIC_COLUMNS)]
                actual = analyzer_a.compare_winrate(analyzer_b, scope)
                assert_series_equal(actual, expected_series, check_exact=False, atol=1e-12)

        self.assertEqual(analyzer_a.compare_winrate(analyzer_b, "horizon")["Bias"], 0.0)
        self.assertEqual(analyzer_a.compare_winrate(analyzer_b, "window")["Bias"], 0.5)

    def test_compare_winrate_requires_matching_indexes(self) -> None:
        analyzer = self._build_analyzer(self.base_raw)

        shifted_dates_raw = self._build_raw(
            self.base_error,
            self.y_true,
            origin_dates=["2024-01-01", "2024-01-03"],
        )
        shifted_dates_analyzer = self._build_analyzer(shifted_dates_raw)
        with self.assertRaisesRegex(ValueError, "different indexes"):
            analyzer.compare_winrate(shifted_dates_analyzer, "window")

        shifted_horizon_raw = self._build_raw(
            self.base_error,
            self.y_true,
            horizons=[1, 2, 4],
        )
        shifted_horizon_analyzer = self._build_analyzer(shifted_horizon_raw)
        with self.assertRaisesRegex(ValueError, "different indexes"):
            analyzer.compare_winrate(shifted_horizon_analyzer, "horizon")

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
        self.assertTrue(np.isfinite(overall["MAPE"]))
        self.assertTrue(np.isfinite(overall["WAPE"]))
        self.assertAlmostEqual(overall["MAPE"], (1.0 / self.eps) * 100, places=2)
        self.assertAlmostEqual(overall["WAPE"], (2.0 / self.eps) * 100, places=2)
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
