from __future__ import annotations

import unittest

from scripts.plot.data import Row, block_runtime_breakdown, infer_solver_families, parse_float_vector, timing_state
from scripts.plot.main import detect_plot_mode, make_plot_specs, partition_plot_specs, resolve_branch_families


class DataTests(unittest.TestCase):
    def test_infer_solver_families_groups_expected_members(self) -> None:
        time_cols = [
            "ns_gupta_batch",
            "ns_ac_fft",
            "ns_discrepancy",
            "ns_mat_update",
            "ns_ac_naive",
            "ns_gupta",
        ]
        families = infer_solver_families(time_cols)
        self.assertEqual(families["ac"], ["ns_ac_naive", "ns_ac_fft"])
        self.assertEqual(families["gupta"], ["ns_gupta", "ns_gupta_batch"])
        self.assertEqual(families["mat_update"], ["ns_mat_update"])

    def test_timing_state_marks_discrepancy_row_invalid(self) -> None:
        fieldnames = [
            "ns_discrepancy",
            "valid_discrepancy",
            "fallback_reason_discrepancy",
        ]
        invalid_row = Row(
            raw={
                "ns_discrepancy": "123",
                "valid_discrepancy": "0",
                "fallback_reason_discrepancy": "fallback",
            },
            values={
                "ns_discrepancy": 123.0,
                "valid_discrepancy": 0.0,
                "fallback_reason_discrepancy": None,
            },
        )
        valid_row = Row(
            raw={
                "ns_discrepancy": "123",
                "valid_discrepancy": "1",
                "fallback_reason_discrepancy": "",
            },
            values={
                "ns_discrepancy": 123.0,
                "valid_discrepancy": 1.0,
                "fallback_reason_discrepancy": None,
            },
        )
        missing_row = Row(
            raw={
                "ns_discrepancy": "",
                "valid_discrepancy": "",
                "fallback_reason_discrepancy": "",
            },
            values={
                "ns_discrepancy": None,
                "valid_discrepancy": None,
                "fallback_reason_discrepancy": None,
            },
        )
        self.assertEqual(timing_state(invalid_row, "ns_discrepancy", fieldnames), "invalid")
        self.assertEqual(timing_state(valid_row, "ns_discrepancy", fieldnames), "usable")
        self.assertEqual(timing_state(missing_row, "ns_discrepancy", fieldnames), "missing")

    def test_timing_state_marks_ac_discrepancy_row_invalid(self) -> None:
        fieldnames = [
            "ns_ac_discrepancy",
            "valid_ac_discrepancy",
            "fallback_reason_ac_discrepancy",
        ]
        invalid_row = Row(
            raw={
                "ns_ac_discrepancy": "123",
                "valid_ac_discrepancy": "0",
                "fallback_reason_ac_discrepancy": "embed_size_cap_base",
            },
            values={
                "ns_ac_discrepancy": 123.0,
                "valid_ac_discrepancy": 0.0,
                "fallback_reason_ac_discrepancy": None,
            },
        )
        self.assertEqual(timing_state(invalid_row, "ns_ac_discrepancy", fieldnames), "invalid")

    def test_make_plot_specs_includes_mat_update_vs_gur_plot(self) -> None:
        specs = make_plot_specs("main", "Bench", {})
        target = next(spec for spec in specs if spec.key == "mat_update_vs_gur_bands")
        self.assertEqual(target.filename, "mat_update_vs_gur_bands.png")
        self.assertEqual(target.kind, "mat_update_vs_gur_bands")
        self.assertTrue(target.x_axis_dependent)
        self.assertEqual(target.solvers, ("mat_update", "gur"))

    def test_make_plot_specs_includes_selected_solver_plot(self) -> None:
        specs = make_plot_specs("main", "Bench", {})
        target = next(spec for spec in specs if spec.key == "selected_solver_bands")
        self.assertEqual(target.filename, "selected_solver_bands.png")
        self.assertEqual(target.kind, "selected_solver_bands")
        self.assertTrue(target.x_axis_dependent)
        self.assertEqual(
            target.solvers,
            ("ac_batch", "ac_fft", "gupta_batch", "gur", "discrepancy"),
        )

    def test_make_plot_specs_includes_selected_solver_detail_plot(self) -> None:
        specs = make_plot_specs("main", "Bench", {})
        target = next(spec for spec in specs if spec.key == "selected_solver_bands_detail")
        self.assertEqual(target.filename, "selected_solver_bands_detail.png")
        self.assertEqual(target.kind, "family_bands_detail")
        self.assertTrue(target.x_axis_dependent)
        self.assertEqual(
            target.solvers,
            ("ac_batch", "ac_fft", "gupta_batch", "gur", "discrepancy"),
        )
        self.assertEqual(target.title, "Bench: selected solver comparison with per-solver points")

    def test_make_plot_specs_keeps_fft_overview_output_contract(self) -> None:
        specs = make_plot_specs("main", "Bench", {})
        target = next(spec for spec in specs if spec.key == "fft_utilization_overview")
        self.assertEqual(target.filename, "fft_utilization_overview.png")
        self.assertEqual(target.kind, "fft_utilization_overview")
        self.assertFalse(target.x_axis_dependent)

    def test_make_plot_specs_acdc_selected_solver_plot_is_isolated(self) -> None:
        specs = make_plot_specs("acdc", "Bench", {})
        target = next(spec for spec in specs if spec.key == "selected_solver_bands")
        self.assertEqual(target.solvers, ("ac_batch", "ac_discrepancy"))
        self.assertEqual(target.kind, "selected_solver_points")
        self.assertTrue(any(spec.key == "acdc_useful_grid_overview" for spec in specs))
        self.assertFalse(any(spec.key == "fft_utilization_overview" for spec in specs))
        self.assertFalse(any(spec.family == "mat_update_vs_gur" for spec in specs))
        self.assertFalse(any(spec.key == "selected_solver_bands_detail" for spec in specs))
        self.assertFalse(any(spec.key == "solver_bands_small_multiples" for spec in specs))
        self.assertFalse(any(spec.key.startswith("branch_") for spec in specs))

    def test_make_plot_specs_omits_default_solver_benchmark_prefix(self) -> None:
        specs = make_plot_specs("main", "Solver benchmark", {"ac": ["ns_ac_batch", "ns_ac_fft"]})
        titles = {spec.key: spec.title for spec in specs}
        self.assertEqual(titles["status_overall"], "overall status counts")
        self.assertEqual(titles["selected_solver_bands"], "selected solver comparison")
        self.assertEqual(titles["branch_ac_bands"], "ac family comparison")

    def test_resolve_branch_families_excludes_ac_base_block_total_metrics(self) -> None:
        families = resolve_branch_families(
            [
                "ns_ac_legacy",
                "ns_ac_naive",
                "ns_ac_batch",
                "ns_ac_fft",
                "ns_ac_batch_base_block_totals",
                "ns_ac_fft_base_block_totals",
            ],
            None,
        )
        self.assertEqual(
            families["ac"],
            ["ns_ac_legacy", "ns_ac_naive", "ns_ac_batch", "ns_ac_fft"],
        )

    def test_make_plot_specs_adds_detail_plot_for_branch_family(self) -> None:
        specs = make_plot_specs("main", "Bench", {"ac": ["ns_ac_batch", "ns_ac_fft"]})
        target = next(spec for spec in specs if spec.key == "branch_ac_bands_detail")
        self.assertEqual(target.filename, "branch_ac_bands_detail.png")
        self.assertEqual(target.kind, "family_bands_detail")
        self.assertEqual(target.solvers, ("ac_batch", "ac_fft"))

    def test_partition_plot_specs_separates_general_and_axis_dependent(self) -> None:
        general, axis = partition_plot_specs(make_plot_specs("main", "Bench", {}))
        self.assertTrue(any(spec.key == "status_overall" for spec in general))
        self.assertTrue(any(spec.key == "fft_utilization_overview" for spec in general))
        self.assertTrue(all(not spec.x_axis_dependent for spec in general))
        self.assertTrue(all(spec.x_axis_dependent for spec in axis))

    def test_detect_plot_mode_prefers_acdc_schema_markers(self) -> None:
        self.assertEqual(detect_plot_mode(["name", "ns_ac_discrepancy"], "auto"), "acdc")
        self.assertEqual(detect_plot_mode(["name", "ns_gur"], "auto"), "main")

    def test_parse_float_vector_parses_semicolon_series(self) -> None:
        self.assertEqual(parse_float_vector("1;2.5;3"), [1.0, 2.5, 3.0])
        self.assertEqual(parse_float_vector(""), [])

    def test_block_runtime_breakdown_extracts_block_kind_shares(self) -> None:
        row = Row(
            raw={"ns_ac_fft_base_block_totals": "10;20;30;40"},
            values={"num_small_blocks": 1.0, "num_big_blocks": 2.0},
            vectors={"ns_ac_fft_base_block_totals": [10.0, 20.0, 30.0, 40.0]},
        )
        breakdown = block_runtime_breakdown(row, "ns_ac_fft_base_block_totals")
        assert breakdown is not None
        self.assertAlmostEqual(breakdown.total, 100.0)
        self.assertAlmostEqual(breakdown.small_share, 0.1)
        self.assertAlmostEqual(breakdown.big_share, 0.5)
        self.assertAlmostEqual(breakdown.slack_share, 0.4)


if __name__ == "__main__":
    unittest.main()

