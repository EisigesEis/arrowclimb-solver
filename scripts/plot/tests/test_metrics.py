from __future__ import annotations

import math
import unittest
from unittest import mock

from scripts.plot.data import Row
from scripts.plot.main import axis_slug, resolve_axis_output_dirs, resolve_x_axes
from scripts.plot.metrics import choose_x_axis, row_proxy_s, row_proxy_s_no_big
from scripts.plot.render import import_matplotlib, render_ac_modeling_cost_overview_plot


class MetricsTests(unittest.TestCase):
    def test_row_proxy_s_matches_formula(self) -> None:
        row = Row(
            raw={},
            values={
                "d": 2.0,
                "p_max": 12.0,
                "Delta": 6.0,
                "nmax": 14.0,
                "mmax": 3.0,
                "ell": 151.0,
                "calN": 16.0,
                "num_big_blocks": 1.0,
                "machine_types": 1.0,
            },
        )
        expected = 2.0 * math.log1p(1.0 * 2.0 * (12.0 + 1.0)) + math.log1p(math.log1p(151.0))
        self.assertAlmostEqual(row_proxy_s(row), expected)

    def test_row_proxy_s_keeps_ell_and_caln_for_small_only_rows(self) -> None:
        row = Row(
            raw={},
            values={
                "d": 2.0,
                "p_max": 12.0,
                "Delta": 6.0,
                "nmax": 14.0,
                "mmax": 3.0,
                "ell": 151.0,
                "calN": 16.0,
                "num_big_blocks": 0.0,
                "machine_types": 1.0,
            },
        )
        expected = 2.0 * math.log1p(1.0 * 2.0 * (12.0 + 1.0)) + math.log1p(math.log1p(151.0))
        self.assertAlmostEqual(row_proxy_s(row), expected)

    def test_row_proxy_s_no_big_ignores_ell_and_caln_for_small_only_rows(self) -> None:
        row = Row(
            raw={},
            values={
                "d": 2.0,
                "p_max": 12.0,
                "Delta": 6.0,
                "nmax": 14.0,
                "mmax": 3.0,
                "ell": 151.0,
                "calN": 16.0,
                "num_big_blocks": 0.0,
                "machine_types": 1.0,
            },
        )
        expected = 2.0 * math.log1p(1.0 * 2.0 * (12.0 + 1.0)) + math.log1p(math.log1p(14.0))
        self.assertAlmostEqual(row_proxy_s_no_big(row), expected)

    def test_choose_x_axis_prefers_proxy_s_when_inputs_exist(self) -> None:
        fieldnames = ["d", "p_max", "Delta", "nmax", "mmax", "ell", "calN", "machine_types", "ns_gur"]
        rows = [
            Row(raw={}, values={"d": 1.0, "p_max": 2.0, "Delta": 1.0, "nmax": 3.0, "mmax": 2.0, "ell": 4.0, "calN": 5.0, "machine_types": 1.0, "ns_gur": 100.0}),
            Row(raw={}, values={"d": 2.0, "p_max": 3.0, "Delta": 1.0, "nmax": 5.0, "mmax": 2.0, "ell": 6.0, "calN": 7.0, "machine_types": 1.0, "ns_gur": 200.0}),
        ]
        self.assertEqual(choose_x_axis(fieldnames, rows, ["ns_gur"], None), "proxy_S")

    def test_choose_x_axis_prefers_proxy_s_with_num_big_blocks_only(self) -> None:
        fieldnames = ["d", "p_max", "Delta", "nmax", "mmax", "num_big_blocks", "machine_types", "ns_gur"]
        rows = [
            Row(raw={}, values={"d": 1.0, "p_max": 2.0, "Delta": 1.0, "nmax": 3.0, "mmax": 2.0, "num_big_blocks": 0.0, "machine_types": 1.0, "ns_gur": 100.0}),
            Row(raw={}, values={"d": 2.0, "p_max": 3.0, "Delta": 1.0, "nmax": 5.0, "mmax": 2.0, "num_big_blocks": 0.0, "machine_types": 1.0, "ns_gur": 200.0}),
        ]
        self.assertEqual(choose_x_axis(fieldnames, rows, ["ns_gur"], None), "p_max")

    def test_choose_x_axis_accepts_proxy_s_no_big_explicitly(self) -> None:
        fieldnames = ["d", "p_max", "Delta", "nmax", "mmax", "num_big_blocks", "machine_types", "ns_gur"]
        rows = [
            Row(raw={}, values={"d": 1.0, "p_max": 2.0, "Delta": 1.0, "nmax": 3.0, "mmax": 2.0, "num_big_blocks": 0.0, "machine_types": 1.0, "ns_gur": 100.0}),
            Row(raw={}, values={"d": 2.0, "p_max": 3.0, "Delta": 1.0, "nmax": 5.0, "mmax": 2.0, "num_big_blocks": 0.0, "machine_types": 1.0, "ns_gur": 200.0}),
        ]
        self.assertEqual(choose_x_axis(fieldnames, rows, ["ns_gur"], "proxy_S_no_big"), "proxy_S_no_big")

    def test_choose_x_axis_falls_back_to_avg_makespan(self) -> None:
        fieldnames = ["name", "avg_makespan", "ns_gur"]
        rows = [
            Row(raw={}, values={"name": None, "avg_makespan": 10.0, "ns_gur": 100.0}),
            Row(raw={}, values={"name": None, "avg_makespan": 20.0, "ns_gur": 200.0}),
        ]
        self.assertEqual(choose_x_axis(fieldnames, rows, ["ns_gur"], None), "avg_makespan")

    def test_resolve_x_axes_supports_comma_selected_variants(self) -> None:
        fieldnames = ["name", "avg_makespan", "p_max", "Delta", "d", "nmax", "mmax", "ell", "calN", "machine_types", "num_big_blocks", "ns_gur"]
        rows = [
            Row(raw={}, values={"avg_makespan": 10.0, "p_max": 2.0, "Delta": 1.0, "d": 1.0, "nmax": 3.0, "mmax": 2.0, "ell": 4.0, "calN": 5.0, "machine_types": 1.0, "num_big_blocks": 1.0, "ns_gur": 100.0}),
        ]
        self.assertEqual(
            resolve_x_axes(fieldnames, rows, ["ns_gur"], ("proxy_S", "proxy_S_no_big", "avg_makespan")),
            ("proxy_S", "proxy_S_no_big", "avg_makespan"),
        )

    def test_axis_slug_sanitizes_names(self) -> None:
        self.assertEqual(axis_slug("avg makespan"), "avg_makespan")

    def test_resolve_axis_output_dirs_disambiguates_slug_collisions(self) -> None:
        dirs = resolve_axis_output_dirs("build/plots", ("avg makespan", "avg_makespan"))
        self.assertNotEqual(dirs["avg makespan"], dirs["avg_makespan"])

    def test_ac_modeling_cost_overview_legend_includes_solver_percentages(self) -> None:
        plt = import_matplotlib()
        overview = {
            "ac_batch": {
                "small_share_median": 0.125,
                "big_share_median": 0.5,
                "slack_share_median": 0.375,
            },
            "ac_discrepancy": {
                "small_share_median": 0.25,
                "big_share_median": 0.625,
                "slack_share_median": 0.125,
            },
            "ac_fft_dummy_work_share_avg": {"values": [], "rows": 0},
        }

        captured: dict[str, object] = {}
        original_close = plt.close

        def capture_close(fig) -> None:
            captured["figure"] = fig

        with mock.patch("matplotlib.figure.Figure.savefig"), mock.patch.object(plt, "close", side_effect=capture_close):
            output_path = "ac_modeling_cost_overview.png"
            render_ac_modeling_cost_overview_plot(plt, overview, "AC modeling", output_path)

        fig = captured["figure"]
        legends = fig.legends
        self.assertEqual(len(legends), 1)
        legend_texts = [text.get_text() for text in legends[0].get_texts()]
        self.assertEqual(
            legend_texts,
            [
                "small (batch 12.5%, discrepancy 25.0%)",
                "big (batch 50.0%, discrepancy 62.5%)",
                "slack (batch 37.5%, discrepancy 12.5%)",
            ],
        )
        original_close(fig)


if __name__ == "__main__":
    unittest.main()
