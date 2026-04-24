from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Optional, Sequence

from .data import (
    BlockRuntimeBreakdown,
    Row,
    block_runtime_breakdown,
    normalize_reason,
    pretty_solver_name,
    solver_family_for_time_col,
    timing_state,
    timing_status_columns_for_time_col,
    unit_scale_to_ms,
)
from .metrics import x_value_for_row


@dataclass
class BinnedSeries:
    time_col: str
    solver: str
    mids: list[float]
    counts: list[int]
    q05: list[Optional[float]]
    q25: list[Optional[float]]
    q50: list[Optional[float]]
    sparse_q50: list[Optional[float]]
    q75: list[Optional[float]]
    q95: list[Optional[float]]
    usable_points: int


def quantile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def make_equal_width_edges(values: Sequence[float], bins: int) -> list[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not finite:
        return [float(index) for index in range(bins + 1)]
    xmin = min(finite)
    xmax = max(finite)
    if xmax <= xmin:
        xmax = xmin + 1.0
    step = (xmax - xmin) / bins
    return [xmin + step * index for index in range(bins + 1)]


def build_solver_points(
    rows: Sequence[Row],
    fieldnames: Sequence[str],
    x_axis: str,
    time_cols: Sequence[str],
    only_success: bool,
) -> dict[str, list[tuple[float, float]]]:
    points_by_solver: dict[str, list[tuple[float, float]]] = {time_col: [] for time_col in time_cols}

    for row_index, row in enumerate(rows):
        if only_success:
            status = row.values.get("status")
            if status is not None and int(status) != 1:
                continue

        x_value = x_value_for_row(row, x_axis, row_index)
        if x_value is None or not math.isfinite(x_value):
            continue

        for time_col in time_cols:
            if timing_state(row, time_col, fieldnames) != "usable":
                continue
            value = row.values.get(time_col)
            if value is None:
                continue
            ms_value = value * unit_scale_to_ms(time_col)
            if not math.isfinite(ms_value) or ms_value <= 0.0:
                continue
            points_by_solver[time_col].append((float(x_value), float(ms_value)))

    return points_by_solver


def build_binned_series(
    points_by_solver: dict[str, list[tuple[float, float]]],
    solver_cols: Sequence[str],
    bins: int,
    min_bin_n: int,
) -> dict[str, BinnedSeries]:
    all_x_values = [x for time_col in solver_cols for x, _ in points_by_solver.get(time_col, [])]
    edges = make_equal_width_edges(all_x_values, bins)
    mids = [(lo + hi) * 0.5 for lo, hi in zip(edges[:-1], edges[1:])]

    series_by_solver: dict[str, BinnedSeries] = {}
    for time_col in solver_cols:
        per_bin: list[list[float]] = [[] for _ in mids]
        for x_value, y_value in points_by_solver.get(time_col, []):
            for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                is_last = index == len(mids) - 1
                if (lo <= x_value < hi) or (is_last and lo <= x_value <= hi):
                    per_bin[index].append(y_value)
                    break

        counts: list[int] = []
        q05: list[Optional[float]] = []
        q25: list[Optional[float]] = []
        q50: list[Optional[float]] = []
        sparse_q50: list[Optional[float]] = []
        q75: list[Optional[float]] = []
        q95: list[Optional[float]] = []

        for values in per_bin:
            counts.append(len(values))
            ordered = sorted(values)
            if len(values) < min_bin_n:
                q05.append(None)
                q25.append(None)
                q50.append(None)
                sparse_q50.append(quantile(ordered, 0.50) if ordered else None)
                q75.append(None)
                q95.append(None)
                continue

            q05.append(quantile(ordered, 0.05))
            q25.append(quantile(ordered, 0.25))
            q50.append(quantile(ordered, 0.50))
            sparse_q50.append(None)
            q75.append(quantile(ordered, 0.75))
            q95.append(quantile(ordered, 0.95))

        series_by_solver[time_col] = BinnedSeries(
            time_col=time_col,
            solver=pretty_solver_name(time_col),
            mids=list(mids),
            counts=counts,
            q05=q05,
            q25=q25,
            q50=q50,
            sparse_q50=sparse_q50,
            q75=q75,
            q95=q95,
            usable_points=len(points_by_solver.get(time_col, [])),
        )
    return series_by_solver


def compute_status_counts(rows: Sequence[Row]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        status = row.values.get("status")
        if status is None:
            counts["missing"] += 1
        else:
            counts[str(int(status))] += 1
    return dict(counts)


def compute_solver_timing_stats(
    rows: Sequence[Row],
    fieldnames: Sequence[str],
    time_cols: Sequence[str],
) -> dict[str, dict[str, object]]:
    total_rows = len(rows)
    summary: dict[str, dict[str, object]] = {}
    for time_col in time_cols:
        usable = 0
        invalid = 0
        missing = 0
        for row in rows:
            state = timing_state(row, time_col, fieldnames)
            if state == "usable":
                usable += 1
            elif state == "invalid":
                invalid += 1
            else:
                missing += 1
        solver = pretty_solver_name(time_col)
        summary[solver] = {
            "time_column": time_col,
            "family": solver_family_for_time_col(time_col),
            "rows_total": total_rows,
            "present": usable + invalid,
            "usable": usable,
            "invalid": invalid,
            "missing": missing,
        }
    return summary


def compute_discrepancy_failure_stats(
    rows: Sequence[Row],
    fieldnames: Sequence[str],
    time_cols: Sequence[str],
) -> dict[str, dict[str, object]]:
    total_rows = len(rows)
    info: dict[str, dict[str, object]] = {}

    for time_col in time_cols:
        solver = pretty_solver_name(time_col)
        if not solver.startswith("discrepancy"):
            continue
        status_cols = timing_status_columns_for_time_col(time_col, fieldnames)
        if status_cols is None:
            continue

        valid = 0
        invalid = 0
        missing = 0
        reasons: Counter[str] = Counter()

        for row in rows:
            raw_value = row.raw.get(time_col, "").strip()
            if not raw_value:
                missing += 1
                continue
            if timing_state(row, time_col, fieldnames) == "invalid":
                invalid += 1
                reason = normalize_reason(row.raw.get(status_cols.reason, "")) if status_cols.reason else "<unknown>"
                reasons[reason] += 1
            else:
                valid += 1

        info[solver] = {
            "rows_total": total_rows,
            "present": valid + invalid,
            "valid": valid,
            "usable": valid,
            "invalid": invalid,
            "missing": missing,
            "failure_rate": (invalid / total_rows) if total_rows else 0.0,
            "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        }

    return info


def compute_ac_block_cost_overview(rows: Sequence[Row]) -> dict[str, dict[str, object]]:
    solver_columns = {
        "ac_batch": "ns_ac_batch_base_block_totals",
        "ac_discrepancy": "ns_ac_discrepancy_base_block_totals",
        "ac_fft": "ns_ac_fft_base_block_totals",
    }
    overview: dict[str, dict[str, object]] = {}

    for solver, column in solver_columns.items():
        breakdowns = [block_runtime_breakdown(row, column) for row in rows]
        usable = [item for item in breakdowns if item is not None]
        overview[solver] = {
            "rows": len(usable),
            "small_share_median": median([item.small_share for item in usable]) if usable else None,
            "big_share_median": median([item.big_share for item in usable]) if usable else None,
            "slack_share_median": median([item.slack_share for item in usable]) if usable else None,
        }

    dummy_values = [
        float(value)
        for row in rows
        for value in [row.values.get("ac_fft_dummy_work_share_avg")]
        if value is not None and math.isfinite(value)
    ]
    overview["ac_fft_dummy_work_share_avg"] = {
        "rows": len(dummy_values),
        "values": [float(value) for value in dummy_values],
    }
    return overview


def compute_acdc_grid_overview(
    rows: Sequence[Row],
    fieldnames: Sequence[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rows": 0,
        "runtime_ms_values": [],
        "useful_grid_share_values": [],
        "rejected_grid_share_values": [],
        "runtime_vs_useful_points": [],
        "stats": {},
    }

    for row in rows:
        if timing_state(row, "ns_ac_discrepancy", fieldnames) != "usable":
            continue

        payload["rows"] = int(payload["rows"]) + 1
        runtime_ns = row.values.get("ns_ac_discrepancy")
        runtime_ms = (
            float(runtime_ns) * unit_scale_to_ms("ns_ac_discrepancy")
            if runtime_ns is not None and math.isfinite(runtime_ns) and runtime_ns > 0.0
            else None
        )
        if runtime_ms is not None:
            payload["runtime_ms_values"].append(runtime_ms)

        useful = row.values.get("ac_discrepancy_useful_grid_share_ns_weighted")
        if useful is None or not math.isfinite(useful):
            continue

        useful_clamped = _clamp_unit(float(useful))
        rejected_clamped = _clamp_unit(1.0 - useful_clamped)
        payload["useful_grid_share_values"].append(useful_clamped)
        payload["rejected_grid_share_values"].append(rejected_clamped)
        if runtime_ms is not None:
            payload["runtime_vs_useful_points"].append(
                {"useful_grid_share": useful_clamped, "runtime_ms": runtime_ms}
            )

    payload["stats"] = {
        key: _summarize_values(value)
        for key, value in payload.items()
        if key.endswith("_values") and isinstance(value, list)
    }
    return payload


def compute_fft_utilization_overview(rows: Sequence[Row]) -> dict[str, list[float]]:
    metric_names = (
        "ac_fft_fft_path_share",
        "ac_fft_fft_padding_efficiency_avg",
        "discrepancy_fft_layer_share",
        "discrepancy_fill_ratio_avg",
    )
    overview: dict[str, list[float]] = {}
    for metric_name in metric_names:
        overview[metric_name] = [
            float(value)
            for row in rows
            for value in [row.values.get(metric_name)]
            if value is not None and math.isfinite(value)
        ]
    return overview


def _clamp_unit(value: float) -> float:
    if value < 0.0 and value > -1e-9:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-9:
        return 1.0
    return max(0.0, min(1.0, value))


def _append_if_finite(values: list[float], raw_value: float | None) -> None:
    if raw_value is None or not math.isfinite(raw_value):
        return
    values.append(float(raw_value))


def _summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(finite),
        "median": median(finite),
        "p10": quantile(finite, 0.10),
        "p90": quantile(finite, 0.90),
    }


def compute_fft_efficiency_summary(
    rows: Sequence[Row],
    fieldnames: Sequence[str],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "definitions": {
            "fft_runtime_share": "Share of convolution/base runtime spent on FFT work instead of sparse or fallback work.",
            "useful_grid_share": "Share of FFT dense grid positions carrying active information.",
            "empty_grid_share": "Share of FFT dense grid positions that were processed but carried no active information.",
            "padding_waste_share": "Share of FFT dense grid positions introduced only by power-of-two padding.",
        },
        "ac_fft": {
            "rows": 0,
            "runtime_ms_values": [],
            "useful_grid_share_values": [],
            "empty_grid_share_values": [],
            "padding_waste_share_values": [],
            "fft_runtime_share_values": [],
            "fft_path_share_values": [],
            "runtime_vs_useful_points": [],
        },
        "discrepancy": {
            "rows": 0,
            "runtime_ms_values": [],
            "useful_grid_share_values": [],
            "empty_grid_share_values": [],
            "padding_waste_share_values": [],
            "fft_runtime_share_values": [],
            "fft_layer_share_values": [],
            "runtime_vs_useful_points": [],
        },
    }

    ac_payload = summary["ac_fft"]
    discrepancy_payload = summary["discrepancy"]
    assert isinstance(ac_payload, dict)
    assert isinstance(discrepancy_payload, dict)

    for row in rows:
        if timing_state(row, "ns_ac_fft", fieldnames) == "usable":
            ac_payload["rows"] = int(ac_payload["rows"]) + 1

            runtime_ns = row.values.get("ns_ac_fft")
            runtime_ms = (
                float(runtime_ns) * unit_scale_to_ms("ns_ac_fft")
                if runtime_ns is not None and math.isfinite(runtime_ns) and runtime_ns > 0.0
                else None
            )
            if runtime_ms is not None:
                ac_payload["runtime_ms_values"].append(runtime_ms)

            useful = row.values.get("ac_fft_useful_grid_share_ns_weighted")
            padding = row.values.get("ac_fft_padding_waste_share_ns_weighted")
            runtime_share = row.values.get("ac_fft_base_runtime_share")
            path_share = row.values.get("ac_fft_fft_path_share")

            if useful is not None and padding is not None and math.isfinite(useful) and math.isfinite(padding):
                useful_clamped = _clamp_unit(float(useful))
                padding_clamped = _clamp_unit(float(padding))
                empty_clamped = _clamp_unit(1.0 - useful_clamped - padding_clamped)
                ac_payload["useful_grid_share_values"].append(useful_clamped)
                ac_payload["padding_waste_share_values"].append(padding_clamped)
                ac_payload["empty_grid_share_values"].append(empty_clamped)
                if runtime_ms is not None:
                    ac_payload["runtime_vs_useful_points"].append(
                        {"useful_grid_share": useful_clamped, "runtime_ms": runtime_ms}
                    )

            if runtime_share is not None and math.isfinite(runtime_share):
                ac_payload["fft_runtime_share_values"].append(_clamp_unit(float(runtime_share)))
            if path_share is not None and math.isfinite(path_share):
                ac_payload["fft_path_share_values"].append(_clamp_unit(float(path_share)))

        if timing_state(row, "ns_discrepancy", fieldnames) == "usable":
            discrepancy_payload["rows"] = int(discrepancy_payload["rows"]) + 1

            runtime_ns = row.values.get("ns_discrepancy")
            runtime_ms = (
                float(runtime_ns) * unit_scale_to_ms("ns_discrepancy")
                if runtime_ns is not None and math.isfinite(runtime_ns) and runtime_ns > 0.0
                else None
            )
            if runtime_ms is not None:
                discrepancy_payload["runtime_ms_values"].append(runtime_ms)

            useful = row.values.get("discrepancy_fft_useful_grid_share_conv_weighted")
            runtime_share = row.values.get("discrepancy_fft_conv_runtime_share")
            layer_share = row.values.get("discrepancy_fft_layer_share")

            if useful is not None and math.isfinite(useful):
                useful_clamped = _clamp_unit(float(useful))
                empty_clamped = _clamp_unit(1.0 - useful_clamped)
                discrepancy_payload["useful_grid_share_values"].append(useful_clamped)
                discrepancy_payload["empty_grid_share_values"].append(empty_clamped)
                if runtime_ms is not None:
                    discrepancy_payload["runtime_vs_useful_points"].append(
                        {"useful_grid_share": useful_clamped, "runtime_ms": runtime_ms}
                    )

            if runtime_share is not None and math.isfinite(runtime_share):
                discrepancy_payload["fft_runtime_share_values"].append(_clamp_unit(float(runtime_share)))
            if layer_share is not None and math.isfinite(layer_share):
                discrepancy_payload["fft_layer_share_values"].append(_clamp_unit(float(layer_share)))

    for solver_name in ("ac_fft", "discrepancy"):
        payload = summary[solver_name]
        assert isinstance(payload, dict)
        payload["stats"] = {
            key: _summarize_values(value)
            for key, value in payload.items()
            if key.endswith("_values") and isinstance(value, list)
        }

    return summary
