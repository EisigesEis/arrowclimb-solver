from __future__ import annotations

import unittest

from scripts.plot.aggregate import (
    build_binned_series,
    compute_ac_block_cost_overview,
    compute_acdc_grid_overview,
    compute_fft_efficiency_summary,
    quantile,
)
from scripts.plot.data import Row


class AggregateTests(unittest.TestCase):
    def test_quantile_linear_interpolation(self) -> None:
        values = [10.0, 20.0]
        self.assertAlmostEqual(quantile(values, 0.5), 15.0)
        self.assertAlmostEqual(quantile(values, 0.05), 10.5)

    def test_build_binned_series_computes_expected_medians(self) -> None:
        points_by_solver = {
            "ns_demo": [
                (0.1, 1.0),
                (0.2, 2.0),
                (0.3, 3.0),
                (0.8, 10.0),
                (0.9, 14.0),
            ]
        }
        series = build_binned_series(points_by_solver, ["ns_demo"], bins=2, min_bin_n=2)["ns_demo"]
        self.assertEqual(series.counts, [3, 2])
        self.assertAlmostEqual(series.q50[0], 2.0)
        self.assertAlmostEqual(series.q25[0], 1.5)
        self.assertAlmostEqual(series.q75[0], 2.5)
        self.assertAlmostEqual(series.q50[1], 12.0)
        self.assertEqual(series.sparse_q50, [None, None])

    def test_build_binned_series_keeps_sparse_bin_median(self) -> None:
        points_by_solver = {
            "ns_demo": [
                (0.1, 1.0),
                (0.2, 2.0),
                (0.3, 3.0),
                (0.8, 10.0),
                (0.9, 14.0),
            ]
        }
        series = build_binned_series(points_by_solver, ["ns_demo"], bins=2, min_bin_n=3)["ns_demo"]
        self.assertAlmostEqual(series.q50[0], 2.0)
        self.assertIsNone(series.q50[1])
        self.assertIsNone(series.sparse_q50[0])
        self.assertAlmostEqual(series.sparse_q50[1], 12.0)

    def test_compute_fft_efficiency_summary_derives_empty_shares(self) -> None:
        rows = [
            Row(
                raw={"ns_ac_fft": "1000", "ns_discrepancy": "2000", "valid_discrepancy": "1"},
                values={
                    "ns_ac_fft": 1000.0,
                    "ac_fft_useful_grid_share_ns_weighted": 0.25,
                    "ac_fft_padding_waste_share_ns_weighted": 0.10,
                    "ac_fft_base_runtime_share": 0.75,
                    "ac_fft_fft_path_share": 1.0,
                    "ns_discrepancy": 2000.0,
                    "discrepancy_fft_useful_grid_share_conv_weighted": 0.2,
                    "discrepancy_fft_conv_runtime_share": 0.9,
                    "discrepancy_fft_layer_share": 1.0,
                    "valid_discrepancy": 1.0,
                },
            )
        ]
        summary = compute_fft_efficiency_summary(rows, ["ns_ac_fft", "ns_discrepancy", "valid_discrepancy"])
        ac = summary["ac_fft"]
        discrepancy = summary["discrepancy"]
        self.assertEqual(ac["rows"], 1)
        self.assertEqual(discrepancy["rows"], 1)
        self.assertAlmostEqual(ac["empty_grid_share_values"][0], 0.65)
        self.assertAlmostEqual(discrepancy["empty_grid_share_values"][0], 0.8)

    def test_compute_fft_efficiency_summary_excludes_invalid_discrepancy_rows(self) -> None:
        rows = [
            Row(
                raw={"ns_discrepancy": "123", "valid_discrepancy": "0", "fallback_reason_discrepancy": "fallback"},
                values={
                    "ns_discrepancy": 123.0,
                    "valid_discrepancy": 0.0,
                    "discrepancy_fft_useful_grid_share_conv_weighted": 0.5,
                    "discrepancy_fft_conv_runtime_share": 1.0,
                    "discrepancy_fft_layer_share": 1.0,
                },
            )
        ]
        summary = compute_fft_efficiency_summary(
            rows,
            ["ns_discrepancy", "valid_discrepancy", "fallback_reason_discrepancy"],
        )
        discrepancy = summary["discrepancy"]
        self.assertEqual(discrepancy["rows"], 0)
        self.assertEqual(discrepancy["runtime_ms_values"], [])
        self.assertEqual(discrepancy["useful_grid_share_values"], [])

    def test_compute_ac_block_cost_overview_keeps_dummy_share_empty_without_profiled_metric(self) -> None:
        rows = [
            Row(
                raw={},
                values={
                    "calN": 4.0,
                    "ac_fft_dummy_work_share_avg": None,
                },
            ),
            Row(
                raw={},
                values={
                    "calN": 0.0,
                    "ac_fft_dummy_work_share_avg": None,
                },
            ),
        ]
        overview = compute_ac_block_cost_overview(rows)
        dummy_payload = overview["ac_fft_dummy_work_share_avg"]
        self.assertEqual(dummy_payload["rows"], 0)
        self.assertEqual(dummy_payload["values"], [])

    def test_compute_acdc_grid_overview_collects_useful_and_rejected_share(self) -> None:
        rows = [
            Row(
                raw={
                    "ns_ac_discrepancy": "1000",
                    "valid_ac_discrepancy": "1",
                },
                values={
                    "ns_ac_discrepancy": 1000.0,
                    "valid_ac_discrepancy": 1.0,
                    "ac_discrepancy_useful_grid_share_ns_weighted": 0.25,
                },
            )
        ]
        overview = compute_acdc_grid_overview(
            rows,
            ["ns_ac_discrepancy", "valid_ac_discrepancy", "ac_discrepancy_useful_grid_share_ns_weighted"],
        )
        self.assertEqual(overview["rows"], 1)
        self.assertAlmostEqual(overview["useful_grid_share_values"][0], 0.25)
        self.assertAlmostEqual(overview["rejected_grid_share_values"][0], 0.75)


if __name__ == "__main__":
    unittest.main()
