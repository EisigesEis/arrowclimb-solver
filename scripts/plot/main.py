from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.plot.aggregate import (  # type: ignore
        build_binned_series,
        build_solver_points,
        compute_ac_block_cost_overview,
        compute_acdc_grid_overview,
        compute_discrepancy_failure_stats,
        compute_fft_efficiency_summary,
        compute_fft_utilization_overview,
        compute_solver_timing_stats,
        compute_status_counts,
    )
    from scripts.plot.cache import (  # type: ignore
        build_manifest,
        build_output_fingerprint,
        file_fingerprint,
        load_manifest,
        should_skip_output,
        write_json,
    )
    from scripts.plot.config import (  # type: ignore
        DEFAULT_BINS,
        DEFAULT_MIN_BIN_N,
        DEFAULT_TITLE_PREFIX,
        PlotSpec,
        RunConfig,
        SINGLE_SOLVER_BLUE,
        SINGLE_SOLVER_ORANGE,
        STATUS_ORDER,
    )
    from scripts.plot.data import infer_solver_families, ordered_time_columns, pretty_solver_name, read_csv_rows  # type: ignore
    from scripts.plot.metrics import choose_x_axis, x_axis_label  # type: ignore
    from scripts.plot.render import (  # type: ignore
        import_matplotlib,
        render_ac_modeling_cost_overview_plot,
        render_acdc_useful_grid_overview_plot,
        render_branch_bands,
        render_branch_bands_detail,
        render_discrepancy_failure_counts_plot,
        render_discrepancy_failure_reasons_plot,
        render_fft_utilization_overview_plot,
        render_solver_point_clouds,
        render_solver_small_multiples,
        render_status_plot,
    )
else:
    from .aggregate import (
        build_binned_series,
        build_solver_points,
        compute_ac_block_cost_overview,
        compute_acdc_grid_overview,
        compute_discrepancy_failure_stats,
        compute_fft_efficiency_summary,
        compute_fft_utilization_overview,
        compute_solver_timing_stats,
        compute_status_counts,
    )
    from .cache import (
        build_manifest,
        build_output_fingerprint,
        file_fingerprint,
        load_manifest,
        should_skip_output,
        write_json,
    )
    from .config import (
        DEFAULT_BINS,
        DEFAULT_MIN_BIN_N,
        DEFAULT_TITLE_PREFIX,
        PlotSpec,
        RunConfig,
        SINGLE_SOLVER_BLUE,
        SINGLE_SOLVER_ORANGE,
        STATUS_ORDER,
    )
    from .data import infer_solver_families, ordered_time_columns, pretty_solver_name, read_csv_rows
    from .metrics import choose_x_axis, x_axis_label
    from .render import (
        import_matplotlib,
        render_ac_modeling_cost_overview_plot,
        render_acdc_useful_grid_overview_plot,
        render_branch_bands,
        render_branch_bands_detail,
        render_discrepancy_failure_counts_plot,
        render_discrepancy_failure_reasons_plot,
        render_fft_utilization_overview_plot,
        render_solver_point_clouds,
        render_solver_small_multiples,
        render_status_plot,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate modular benchmark plots from main.csv-like data.")
    parser.add_argument("--input", default=os.path.join("instances", "main.csv"), help="Input CSV path")
    parser.add_argument("--out-dir", default=os.path.join("build", "plots"), help="Directory for PNG/JSON outputs")
    parser.add_argument("--mode", choices=("auto", "main", "acdc"), default="auto", help="Plot mode selection")
    parser.add_argument("--x-axis", default=None, help="Comma-separated x-axis column names or registered metric names")
    parser.add_argument("--only-success", action="store_true", help="Only include rows with status == 1 for runtime plots")
    parser.add_argument("--title-prefix", default=DEFAULT_TITLE_PREFIX, help="Prefix for plot titles")
    parser.add_argument("--force", action="store_true", help="Regenerate all outputs and ignore manifest-based skips")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help=f"Number of shared x bins (default: {DEFAULT_BINS})")
    parser.add_argument(
        "--min-bin-n",
        type=int,
        default=DEFAULT_MIN_BIN_N,
        help=f"Minimum points required per bin (default: {DEFAULT_MIN_BIN_N})",
    )
    parser.add_argument(
        "--families",
        default=None,
        help="Optional comma-separated branch families to plot (default: all non-singleton families)",
    )
    return parser.parse_args(argv)


def _parse_csv_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or None


def _parse_families(raw: str | None) -> tuple[str, ...] | None:
    return _parse_csv_list(raw)


def _parse_x_axes(raw: str | None) -> tuple[str, ...] | None:
    return _parse_csv_list(raw)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if args.min_bin_n <= 0:
        raise ValueError("--min-bin-n must be positive")
    return RunConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        mode=args.mode,
        x_axes=_parse_x_axes(args.x_axis),
        only_success=bool(args.only_success),
        title_prefix=args.title_prefix,
        force=bool(args.force),
        bins=int(args.bins),
        min_bin_n=int(args.min_bin_n),
        families=_parse_families(args.families),
        out_dir_is_default=(args.out_dir == os.path.join("build", "plots")),
    )


BRANCH_FAMILY_EXCLUDED_TIME_COLS = frozenset(
    {
        "ns_ac_batch_base_block_totals",
        "ns_ac_discrepancy_base_block_totals",
        "ns_ac_fft_base_block_totals",
    }
)


def resolve_branch_families(
    time_cols: list[str],
    requested_families: tuple[str, ...] | None,
) -> dict[str, list[str]]:
    families = infer_solver_families(time_cols)
    branch_families = {
        family: [
            member for member in members if member not in BRANCH_FAMILY_EXCLUDED_TIME_COLS
        ]
        for family, members in families.items()
        if family not in {"mat_update", "gur"}
        and len([member for member in members if member not in BRANCH_FAMILY_EXCLUDED_TIME_COLS]) >= 2
    }
    if requested_families is None:
        return branch_families

    requested = {}
    for family in requested_families:
        if family not in branch_families:
            raise ValueError(f"Requested family '{family}' is unavailable or not comparable in this CSV")
        requested[family] = branch_families[family]
    return requested


def resolve_x_axes(
    fieldnames: list[str],
    rows,
    time_cols: list[str],
    requested_x_axes: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested_x_axes is None:
        return (choose_x_axis(fieldnames, rows, time_cols, None),)

    resolved: list[str] = []
    for requested in requested_x_axes:
        axis = choose_x_axis(fieldnames, rows, time_cols, requested)
        if axis not in resolved:
            resolved.append(axis)
    return tuple(resolved)


def axis_slug(axis: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", axis).strip("._-").lower()
    return slug or "axis"


def resolve_axis_output_dirs(out_dir: str, axes: tuple[str, ...]) -> dict[str, str]:
    axis_dirs: dict[str, str] = {}
    used: dict[str, str] = {}
    for axis in axes:
        base_slug = axis_slug(axis)
        slug = base_slug
        if slug in used and used[slug] != axis:
            slug = f"{base_slug}_{hashlib.sha1(axis.encode('utf-8')).hexdigest()[:8]}"
        used[slug] = axis
        axis_dirs[axis] = os.path.join(out_dir, slug)
    return axis_dirs


def detect_plot_mode(fieldnames: list[str], requested_mode: str) -> str:
    if requested_mode in {"main", "acdc"}:
        return requested_mode
    header = set(fieldnames)
    if "ns_ac_discrepancy" in header or "valid_ac_discrepancy" in header:
        return "acdc"
    return "main"


def default_out_dir_for_mode(mode: str) -> str:
    return os.path.join("build", "plots_acdc" if mode == "acdc" else "plots")


def plot_title(title_prefix: str, label: str) -> str:
    clean_prefix = title_prefix.strip()
    if not clean_prefix or clean_prefix == DEFAULT_TITLE_PREFIX:
        return label
    return f"{clean_prefix}: {label}"


def comparison_title(title_prefix: str, label: str, *, detail: bool = False) -> str:
    suffix = " with per-solver points" if detail else ""
    return plot_title(title_prefix, f"{label}{suffix}")


def family_comparison_label(family: str) -> str:
    if family == "selected_solver_bands":
        return "selected solver comparison"
    if family == "mat_update_vs_gur":
        return "mat_update vs gur comparison"
    return f"{family} family comparison"


def make_plot_specs(mode: str, title_prefix: str, branch_families: dict[str, list[str]]) -> list[PlotSpec]:
    specs = [
        PlotSpec(
            key="status_overall",
            filename="status_overall.png",
            kind="status_overall",
            title=plot_title(title_prefix, "overall status counts"),
            x_axis_dependent=False,
        ),
        PlotSpec(
            key="ac_modeling_cost_overview",
            filename="ac_modeling_cost_overview.png",
            kind="ac_modeling_cost_overview",
            title=plot_title(title_prefix, "AC modeling cost overview"),
            x_axis_dependent=False,
        ),
        PlotSpec(
            key="mat_update_vs_gur_bands",
            filename="mat_update_vs_gur_bands.png",
            kind="mat_update_vs_gur_bands",
            title=comparison_title(title_prefix, family_comparison_label("mat_update_vs_gur")),
            x_axis_dependent=True,
            family="mat_update_vs_gur",
            solvers=("mat_update", "gur"),
        ),
        PlotSpec(
            key="mat_update_vs_gur_bands_detail",
            filename="mat_update_vs_gur_bands_detail.png",
            kind="family_bands_detail",
            title=comparison_title(title_prefix, family_comparison_label("mat_update_vs_gur"), detail=True),
            x_axis_dependent=True,
            family="mat_update_vs_gur",
            solvers=("mat_update", "gur"),
        ),
        PlotSpec(
            key="selected_solver_bands",
            filename="selected_solver_bands.png",
            kind="selected_solver_bands",
            title=comparison_title(title_prefix, family_comparison_label("selected_solver_bands")),
            x_axis_dependent=True,
            family="selected_solver_bands",
            solvers=("ac_batch", "ac_discrepancy")
            if mode == "acdc"
            else ("ac_batch", "ac_fft", "gupta_batch", "gur", "discrepancy"),
        ),
    ]
    specs.insert(
        2,
        PlotSpec(
            key="solver_bands_small_multiples",
            filename="solver_bands_small_multiples.png",
            kind="solver_bands_small_multiples",
            title=plot_title(title_prefix, "solver runtime bands"),
            x_axis_dependent=True,
        ),
    )
    if mode != "acdc":
        specs.append(
            PlotSpec(
                key="selected_solver_bands_detail",
                filename="selected_solver_bands_detail.png",
                kind="family_bands_detail",
                title=comparison_title(title_prefix, family_comparison_label("selected_solver_bands"), detail=True),
                x_axis_dependent=True,
                family="selected_solver_bands",
                solvers=("ac_batch", "ac_fft", "gupta_batch", "gur", "discrepancy"),
            )
        )
    if mode == "main":
        specs[1:1] = [
            PlotSpec(
                key="discrepancy_failure_counts",
                filename="discrepancy_failure_counts.png",
                kind="discrepancy_failure_counts",
                title=plot_title(title_prefix, "discrepancy timing validity"),
                x_axis_dependent=False,
            ),
            PlotSpec(
                key="discrepancy_failure_reasons",
                filename="discrepancy_failure_reasons.png",
                kind="discrepancy_failure_reasons",
                title=plot_title(title_prefix, "discrepancy invalidation reasons"),
                x_axis_dependent=False,
            ),
            PlotSpec(
                key="fft_utilization_overview",
                filename="fft_utilization_overview.png",
                kind="fft_utilization_overview",
                title=plot_title(title_prefix, "FFT used-grid-share vs cost overview"),
                x_axis_dependent=False,
            ),
        ]
    else:
        specs = [
            replace(plot_spec, kind="selected_solver_points")
            if plot_spec.key == "selected_solver_bands"
            else plot_spec
            for plot_spec in specs
        ]
        specs.insert(
            2,
            PlotSpec(
                key="acdc_useful_grid_overview",
                filename="acdc_useful_grid_overview.png",
                kind="acdc_useful_grid_overview",
                title=plot_title(title_prefix, "ac_discrepancy useful grid overview"),
                x_axis_dependent=False,
            ),
        )
        specs = [
            plot_spec
            for plot_spec in specs
            if plot_spec.kind != "mat_update_vs_gur_bands"
            and plot_spec.family != "mat_update_vs_gur"
            and plot_spec.kind != "solver_bands_small_multiples"
        ]
        return specs

    for family, members in branch_families.items():
        specs.append(
            PlotSpec(
                key=f"branch_{family}_bands",
                filename=f"branch_{family}_bands.png",
                kind="branch_bands",
                title=comparison_title(title_prefix, family_comparison_label(family)),
                x_axis_dependent=True,
                family=family,
                solvers=tuple(pretty_solver_name(member) for member in members),
            )
        )
        specs.append(
            PlotSpec(
                key=f"branch_{family}_bands_detail",
                filename=f"branch_{family}_bands_detail.png",
                kind="family_bands_detail",
                title=comparison_title(title_prefix, family_comparison_label(family), detail=True),
                x_axis_dependent=True,
                family=family,
                solvers=tuple(pretty_solver_name(member) for member in members),
            )
        )
    return specs


def partition_plot_specs(plot_specs: list[PlotSpec]) -> tuple[list[PlotSpec], list[PlotSpec]]:
    general = [plot_spec for plot_spec in plot_specs if not plot_spec.x_axis_dependent]
    axis_dependent = [plot_spec for plot_spec in plot_specs if plot_spec.x_axis_dependent]
    return general, axis_dependent


def cleanup_legacy_outputs(out_dir: str, active_filenames: set[str], force: bool) -> None:
    if not force:
        return
    legacy_filenames = {
        "medians_overlay_with_mat_update.png",
        "medians_overlay_without_mat_update.png",
        "medians_small_multiples.png",
        "selected_solver_bands_detail.png",
    }
    for filename in legacy_filenames - active_filenames:
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    for filename in os.listdir(out_dir):
        if filename in active_filenames:
            continue
        if filename.startswith("branch_") and (
            filename.endswith("_bands.png") or filename.endswith("_bands_detail.png")
        ):
            path = os.path.join(out_dir, filename)
            if os.path.isfile(path):
                os.remove(path)


def render_general_plot(
    plt,
    plot_spec: PlotSpec,
    output_path: str,
    mode: str,
    status_counts: dict[str, int],
    discrepancy_info: dict[str, dict[str, object]],
    ac_block_cost_overview: dict[str, dict[str, object]],
    fft_efficiency_summary: dict[str, object],
    acdc_grid_overview: dict[str, object],
) -> None:
    if plot_spec.kind == "status_overall":
        render_status_plot(plt, status_counts, plot_spec.title, output_path, STATUS_ORDER)
    elif plot_spec.kind == "discrepancy_failure_counts":
        render_discrepancy_failure_counts_plot(plt, discrepancy_info, plot_spec.title, output_path)
    elif plot_spec.kind == "discrepancy_failure_reasons":
        render_discrepancy_failure_reasons_plot(plt, discrepancy_info, plot_spec.title, output_path)
    elif plot_spec.kind == "ac_modeling_cost_overview":
        render_ac_modeling_cost_overview_plot(plt, ac_block_cost_overview, plot_spec.title, output_path)
    elif plot_spec.kind == "fft_utilization_overview":
        render_fft_utilization_overview_plot(plt, fft_efficiency_summary, plot_spec.title, output_path)
    elif plot_spec.kind == "acdc_useful_grid_overview":
        render_acdc_useful_grid_overview_plot(plt, acdc_grid_overview, plot_spec.title, output_path)
    else:
        raise ValueError(f"Unknown general plot kind: {plot_spec.kind}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_run_config(args)

    fieldnames, rows = read_csv_rows(config.input_path)
    resolved_mode = detect_plot_mode(fieldnames, config.mode)
    out_dir = default_out_dir_for_mode(resolved_mode) if config.out_dir_is_default else config.out_dir
    config = replace(config, mode=resolved_mode, out_dir=out_dir, out_dir_is_default=False)
    time_cols = ordered_time_columns(fieldnames)
    if not time_cols:
        raise ValueError("No time columns found (expected prefixes ns_/us_/ms_/s_).")

    resolved_axes = resolve_x_axes(fieldnames, rows, time_cols, config.x_axes)
    axis_output_dirs = resolve_axis_output_dirs(config.out_dir, resolved_axes)
    branch_families = resolve_branch_families(time_cols, config.families)

    status_counts = compute_status_counts(rows)
    solver_timing_stats = compute_solver_timing_stats(rows, fieldnames, time_cols)
    discrepancy_info = compute_discrepancy_failure_stats(rows, fieldnames, time_cols)
    ac_block_cost_overview = compute_ac_block_cost_overview(rows)
    fft_utilization_overview = compute_fft_utilization_overview(rows)
    fft_efficiency_summary = compute_fft_efficiency_summary(rows, fieldnames)
    acdc_grid_overview = compute_acdc_grid_overview(rows, fieldnames)

    os.makedirs(config.out_dir, exist_ok=True)
    manifest_path = os.path.join(config.out_dir, "manifest.json")
    summary_path = os.path.join(config.out_dir, "summary.json")
    existing_manifest = load_manifest(manifest_path)
    input_fp = file_fingerprint(config.input_path)
    plot_specs = make_plot_specs(config.mode, config.title_prefix, branch_families)
    general_specs, axis_specs = partition_plot_specs(plot_specs)
    cleanup_legacy_outputs(config.out_dir, {plot_spec.filename for plot_spec in general_specs}, config.force)
    for axis_dir in axis_output_dirs.values():
        os.makedirs(axis_dir, exist_ok=True)
        cleanup_legacy_outputs(axis_dir, {plot_spec.filename for plot_spec in axis_specs}, config.force)
    plt = import_matplotlib()

    output_records: dict[str, dict[str, object]] = {}
    summary_outputs: dict[str, dict[str, object]] = {}

    for plot_spec in general_specs:
        output_path = os.path.join(config.out_dir, plot_spec.filename)
        fingerprint = build_output_fingerprint(input_fp, config, plot_spec)
        scoped_key = f"general:{plot_spec.key}"
        skipped = should_skip_output(output_path, existing_manifest, scoped_key, fingerprint, config.force)

        if not skipped:
            render_general_plot(
                plt,
                plot_spec,
                output_path,
                config.mode,
                status_counts,
                discrepancy_info,
                ac_block_cost_overview,
                fft_efficiency_summary,
                acdc_grid_overview,
            )

        output_records[scoped_key] = {
            "path": output_path,
            "fingerprint": fingerprint,
            "kind": plot_spec.kind,
            "scope": "general",
            "family": plot_spec.family,
            "solvers": list(plot_spec.solvers),
        }
        summary_outputs[scoped_key] = {
            "path": output_path,
            "state": "skipped" if skipped else "generated",
        }

    mat_update_vs_gur_cols = [
        time_col for time_col in time_cols if pretty_solver_name(time_col) in {"mat_update", "gur"}
    ]
    selected_solver_names = (
        {"ac_batch", "ac_discrepancy"}
        if config.mode == "acdc"
        else {"ac_batch", "ac_fft", "gupta_batch", "gur", "discrepancy"}
    )
    selected_solver_cols = [
        time_col for time_col in time_cols if pretty_solver_name(time_col) in selected_solver_names
    ]

    for x_axis in resolved_axes:
        x_label = x_axis_label(x_axis)
        axis_dir = axis_output_dirs[x_axis]
        points_by_solver = build_solver_points(rows, fieldnames, x_axis, time_cols, config.only_success)
        solver_bands = build_binned_series(points_by_solver, time_cols, config.bins, config.min_bin_n)
        mat_update_vs_gur_bands = build_binned_series(
            points_by_solver,
            mat_update_vs_gur_cols,
            config.bins,
            config.min_bin_n,
        )
        selected_solver_bands = build_binned_series(
            points_by_solver,
            selected_solver_cols,
            config.bins,
            config.min_bin_n,
        )
        branch_bands = {
            family: build_binned_series(points_by_solver, members, config.bins, config.min_bin_n)
            for family, members in branch_families.items()
        }

        for plot_spec in axis_specs:
            output_path = os.path.join(axis_dir, plot_spec.filename)
            fingerprint = build_output_fingerprint(input_fp, config, plot_spec)
            scoped_key = f"axis:{x_axis}:{plot_spec.key}"
            skipped = should_skip_output(output_path, existing_manifest, scoped_key, fingerprint, config.force)

            if not skipped:
                if plot_spec.kind == "solver_bands_small_multiples":
                    render_solver_small_multiples(
                        plt,
                        {time_col: solver_bands[time_col] for time_col in time_cols if time_col in solver_bands},
                        x_label,
                        plot_spec.title,
                        output_path,
                        mat_update_color=SINGLE_SOLVER_BLUE,
                        other_color=SINGLE_SOLVER_ORANGE,
                    )
                elif plot_spec.kind == "selected_solver_points":
                    render_solver_point_clouds(
                        plt,
                        {time_col: points_by_solver.get(time_col, []) for time_col in selected_solver_cols},
                        x_label,
                        plot_spec.title,
                        output_path,
                    )
                elif plot_spec.kind == "mat_update_vs_gur_bands":
                    render_branch_bands(
                        plt,
                        plot_spec.family or "mat_update_vs_gur",
                        mat_update_vs_gur_bands,
                        x_label,
                        plot_spec.title,
                        output_path,
                    )
                elif plot_spec.kind == "selected_solver_bands":
                    render_branch_bands(
                        plt,
                        plot_spec.family or "selected_solver_bands",
                        selected_solver_bands,
                        x_label,
                        plot_spec.title,
                        output_path,
                    )
                elif plot_spec.kind == "family_bands_detail":
                    if plot_spec.family == "mat_update_vs_gur":
                        detail_series = mat_update_vs_gur_bands
                    elif plot_spec.family == "selected_solver_bands":
                        detail_series = selected_solver_bands
                    else:
                        detail_series = branch_bands.get(plot_spec.family or "", {})
                    render_branch_bands_detail(
                        plt,
                        plot_spec.family or "family",
                        detail_series,
                        {time_col: points_by_solver.get(time_col, []) for time_col in detail_series},
                        x_label,
                        plot_spec.title,
                        output_path,
                    )
                elif plot_spec.kind == "branch_bands" and plot_spec.family is not None:
                    render_branch_bands(
                        plt,
                        plot_spec.family,
                        branch_bands.get(plot_spec.family, {}),
                        x_label,
                        plot_spec.title,
                        output_path,
                    )
                else:
                    raise ValueError(f"Unknown axis plot kind: {plot_spec.kind}")

            output_records[scoped_key] = {
                "path": output_path,
                "fingerprint": fingerprint,
                "kind": plot_spec.kind,
                "scope": "axis",
                "x_axis": x_axis,
                "family": plot_spec.family,
                "solvers": list(plot_spec.solvers),
            }
            summary_outputs[scoped_key] = {
                "path": output_path,
                "state": "skipped" if skipped else "generated",
            }

    manifest = build_manifest(config.input_path, input_fp, config, output_records)
    write_json(manifest_path, manifest)

    summary = {
        "input": {
            "path": config.input_path,
            "fingerprint": input_fp,
        },
        "mode": config.mode,
        "rows": len(rows),
        "x_axes": [
            {
                "name": x_axis,
                "label": x_axis_label(x_axis),
                "output_dir": axis_output_dirs[x_axis],
            }
            for x_axis in resolved_axes
        ],
        "only_success_for_runtime_plots": config.only_success,
        "bins": config.bins,
        "min_bin_n": config.min_bin_n,
        "time_columns": time_cols,
        "branch_families": {
            family: [pretty_solver_name(member) for member in members]
            for family, members in branch_families.items()
        },
        "status_counts": status_counts,
        "solver_timing_stats": solver_timing_stats,
        "discrepancy_failures": discrepancy_info,
        "ac_block_cost_overview": ac_block_cost_overview,
        "fft_utilization_overview": {
            metric_name: {
                "rows": len(values),
            }
            for metric_name, values in fft_utilization_overview.items()
        },
        "fft_efficiency_summary": fft_efficiency_summary,
        "acdc_grid_overview": acdc_grid_overview,
        "outputs": summary_outputs,
        "manifest": manifest_path,
    }
    write_json(summary_path, summary)

    print(f"Input:      {config.input_path}")
    print(f"Rows:       {len(rows)}")
    print(f"X-axes:     {', '.join(resolved_axes)}")
    print(f"Out dir:    {config.out_dir}")
    print("Outputs:")
    for key, payload in summary_outputs.items():
        print(f"  {key}: {payload['state']}")
    print(f"Summary:    {summary_path}")
    print(f"Manifest:   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
