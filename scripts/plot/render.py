from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from .aggregate import BinnedSeries
from .config import (
    FAMILY_LINESTYLES,
    FAMILY_PALETTE,
    INNER_BAND_ALPHA,
    MEDIAN_LINEWIDTH,
    OUTER_BAND_ALPHA,
)
from .data import pretty_solver_name


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "matplotlib is required for plotting.\n"
                "Install it with: python -m pip install matplotlib"
            ) from None
        raise
    return plt


def _finalize_figure(fig, *, top: float = 0.9, right: float = 0.98) -> None:
    fig.tight_layout(rect=(0.0, 0.0, right, top))


def _median(values: Sequence[float]) -> Optional[float]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2 == 1:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return None
    if q <= 0.0:
        return finite[0]
    if q >= 1.0:
        return finite[-1]
    position = q * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _positive_values_log_worthy(values: Sequence[float]) -> bool:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value)) and value > 0.0]
    if not finite:
        return False
    return max(finite) / min(finite) > 25.0


def _values_log_worthy(values: Sequence[float]) -> bool:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value)) and value > 0.0]
    if not finite:
        return False
    return max(finite) / max(min(finite), 1e-12) > 20.0


def _format_pct(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _share_legend_label(kind: str, solver_names: Sequence[str], shares: Sequence[Optional[float]]) -> str:
    parts = [
        f"{solver.replace('ac_', '')} {_format_pct(share)}"
        for solver, share in zip(solver_names, shares)
    ]
    return f"{kind} ({', '.join(parts)})"


def _x_limits_with_padding(xs: Sequence[float], *, log_scale: bool) -> tuple[float, float]:
    finite = [float(value) for value in xs if value is not None and math.isfinite(float(value))]
    if not finite:
        return (0.0, 1.0)
    lower = min(finite)
    upper = max(finite)
    if log_scale:
        positive = [value for value in finite if value > 0.0]
        if not positive:
            return (0.0, 1.0)
        lower = min(positive)
        upper = max(positive)
        if upper <= lower:
            factor = 1.15
            return (lower / factor, upper * factor)
        return (lower / 1.15, upper * 1.15)

    span = max(upper - lower, 1e-9)
    pad = span * 0.04
    return (max(0.0, lower - pad), upper + pad)


def _runtime_y_limits_with_padding(ys: Sequence[float], *, log_scale: bool) -> tuple[float, float]:
    finite = [float(value) for value in ys if value is not None and math.isfinite(float(value))]
    if not finite:
        return (0.0, 1.0)
    lower = min(finite)
    upper = max(finite)
    if log_scale:
        positive = [value for value in finite if value > 0.0]
        if not positive:
            return (0.0, 1.0)
        lower = min(positive)
        if upper <= lower:
            factor = 1.15
            return (lower / factor, lower * factor)
        return (lower / 1.15, upper * 1.15)

    span = max(upper - lower, 1e-9)
    pad = span * 0.04
    return (max(0.0, lower - pad), upper + pad)


def _visible_hexbin_y_limits(hexbin_artist, ys: Sequence[float], *, log_scale: bool) -> tuple[tuple[float, float], float]:
    base_limits = _runtime_y_limits_with_padding(ys, log_scale=log_scale)
    offsets = getattr(hexbin_artist, "get_offsets", lambda: [])()
    if offsets is None or len(offsets) == 0:
        return (base_limits, base_limits[1])
    counts = getattr(hexbin_artist, "get_array", lambda: [])()

    lower = min(float(value) for value in ys if value is not None and math.isfinite(float(value)) and (not log_scale or float(value) > 0.0))
    if log_scale:
        log_rows = [
            (float(point[1]), float(count))
            for point, count in zip(offsets, counts)
            if len(point) >= 2 and math.isfinite(float(point[1])) and math.isfinite(float(count))
        ]
        if not log_rows:
            return (base_limits, base_limits[1])
        dense_log_y = [value for value, count in log_rows if count >= 2.0]
        log_y = dense_log_y if dense_log_y else [value for value, _ in log_rows]
        half_step = 0.0
        if len(log_y) >= 2:
            unique_log_y = sorted(set(log_y))
            positive_steps = [b - a for a, b in zip(unique_log_y, unique_log_y[1:]) if b > a]
            if positive_steps:
                half_step = min(positive_steps) * 0.5
        visible_upper = 10 ** max(log_y)
        q99 = _quantile(
            [float(value) for value in ys if value is not None and math.isfinite(float(value)) and float(value) > 0.0],
            0.992,
        )
        if q99 is not None:
            visible_upper = max(visible_upper, q99)
        upper = 10 ** (max(log_y) + half_step)
        if q99 is not None:
            upper = max(upper, q99 * 1.08)
        if upper <= lower:
            factor = 1.15
            return ((lower / factor, lower * factor), lower)
        return ((lower / 1.15, upper * 1.15), visible_upper)

    visible_y = [
        (float(point[1]), float(count))
        for point, count in zip(offsets, counts)
        if len(point) >= 2 and math.isfinite(float(point[1])) and math.isfinite(float(count))
    ]
    if not visible_y:
        return (base_limits, base_limits[1])
    dense_y = [value for value, count in visible_y if count >= 2.0]
    upper = max(dense_y) if dense_y else max(value for value, _ in visible_y)
    span = max(upper - lower, 1e-9)
    pad = span * 0.04
    return ((max(0.0, lower - pad), upper + pad), upper)


def _draw_clipped_outlier_markers(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    upper_visible: float,
    y_log: bool,
    color: str,
) -> int:
    clipped = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y)) and float(y) > upper_visible
    ]
    if not clipped:
        return 0
    y_top = ax.get_ylim()[1]
    marker_y = math.sqrt(upper_visible * y_top) if y_log else 0.5 * (upper_visible + y_top)
    ax.scatter(
        [x for x, _ in clipped],
        [marker_y] * len(clipped),
        marker="o",
        s=18,
        linewidths=0.9,
        facecolors="none",
        edgecolors=color,
        alpha=0.9,
        clip_on=True,
        zorder=4,
    )
    return len(clipped)


def _solver_panel_stats(payload: dict[str, object], *, include_padding: bool, clipped_singles: int = 0) -> str:
    useful = payload.get("useful_grid_share_values", [])
    empty = payload.get("empty_grid_share_values", [])
    lines = [
        f"n={int(payload.get('rows', 0) or 0)}",
        f"mean useful: {_format_pct(_mean(useful if isinstance(useful, list) else []))}",
        f"mean empty: {_format_pct(_mean(empty if isinstance(empty, list) else []))}",
    ]
    if include_padding:
        padding = payload.get("padding_waste_share_values", [])
        lines.append(f"mean padding: {_format_pct(_mean(padding if isinstance(padding, list) else []))}")
    if clipped_singles > 0:
        lines.append(f"clipped singles: {clipped_singles}")
    return "\n".join(lines)


def _render_runtime_vs_useful_scatter(
    ax,
    payload: dict[str, object],
    title: str,
    density_cmap: str,
    *,
    include_padding: bool,
    share_label: str = "useful grid share",
    stats_box_position: str = "top_right",
) -> None:
    points = payload.get("runtime_vs_useful_points", [])
    xs = [
        float(point["useful_grid_share"])
        for point in points
        if isinstance(point, dict)
        and point.get("useful_grid_share") is not None
        and point.get("runtime_ms") is not None
        and math.isfinite(float(point["useful_grid_share"]))
        and math.isfinite(float(point["runtime_ms"]))
    ]
    ys = [
        float(point["runtime_ms"])
        for point in points
        if isinstance(point, dict)
        and point.get("useful_grid_share") is not None
        and point.get("runtime_ms") is not None
        and math.isfinite(float(point["useful_grid_share"]))
        and math.isfinite(float(point["runtime_ms"]))
    ]
    if not xs or not ys:
        _render_empty_panel(ax, title, "No usable runtime-vs-grid-share data found.")
        return

    x_log = _positive_values_log_worthy(xs)
    y_log = _values_log_worthy(ys)
    hexbin_kwargs = {
        "gridsize": (54, 34) if x_log else (42, 34),
        "mincnt": 1,
        "linewidths": 0.0,
        "bins": "log",
        "cmap": density_cmap,
    }
    if x_log:
        hexbin_kwargs["xscale"] = "log"
    if y_log:
        hexbin_kwargs["yscale"] = "log"
    hexbin_artist = ax.hexbin(xs, ys, **hexbin_kwargs)

    if x_log:
        ax.set_xscale("log")
        ax.set_xlim(*_x_limits_with_padding(xs, log_scale=True))
        ax.set_xlabel(f"{share_label} (log scale)")
    else:
        ax.set_xlim(*_x_limits_with_padding(xs, log_scale=False))
        ax.set_xlabel(share_label)

    y_limits, upper_visible = _visible_hexbin_y_limits(hexbin_artist, ys, log_scale=y_log)
    if y_log:
        ax.set_yscale("log")
        ax.set_ylim(*y_limits)
        ax.set_ylabel("runtime (ms, log scale)")
    else:
        ax.set_ylim(*y_limits)
        ax.set_ylabel("runtime (ms)")
    clipped_singles = _draw_clipped_outlier_markers(
        ax,
        xs,
        ys,
        upper_visible=upper_visible,
        y_log=y_log,
        color="#355C7D" if density_cmap == "Blues" else "#2E7D32",
    )
    loc, bbox_anchor = {
        "bottom_left": ("lower left", (0.010, 0.015)),
        "bottom_right": ("lower right", (0.965, 0.035)),
        "top_right": ("upper right", (0.965, 0.965)),
    }.get(stats_box_position, ("upper right", (0.965, 0.965)))
    ax.set_facecolor("#FBFBFB")
    ax.set_title(title)
    ax.grid(True, axis="both", alpha=0.25)
    from matplotlib.offsetbox import AnchoredText

    stats_box = AnchoredText(
        _solver_panel_stats(payload, include_padding=include_padding, clipped_singles=clipped_singles),
        loc=loc,
        prop={"size": 9},
        frameon=True,
        pad=0.35,
        borderpad=0.28,
        bbox_to_anchor=bbox_anchor,
        bbox_transform=ax.transAxes,
    )
    stats_box.patch.set_boxstyle("round,pad=0.35")
    stats_box.patch.set_facecolor("white")
    stats_box.patch.set_edgecolor("#B8B8B8")
    stats_box.patch.set_alpha(0.9)
    ax.add_artist(stats_box)


def _draw_component_barh(ax, label: str, segments: list[tuple[str, float, str]]) -> None:
    left = 0.0
    for segment_label, value, color in segments:
        if value <= 0.0:
            continue
        ax.barh([label], [value], left=left, color=color, edgecolor="white", linewidth=0.8)
        left += value


def _safe_solver_payload(overview: dict[str, object], key: str) -> dict[str, object]:
    payload = overview.get(key, {})
    return payload if isinstance(payload, dict) else {}


def _series_values(series: Iterable[BinnedSeries]) -> list[float]:
    values: list[float] = []
    for item in series:
        for bucket in (item.q05, item.q25, item.q50, item.q75, item.q95):
            values.extend(value for value in bucket if value is not None and value > 0.0)
    return values


def y_is_log_worthy(series: Iterable[BinnedSeries]) -> bool:
    values = _series_values(series)
    if not values:
        return False
    return max(values) / max(min(values), 1e-12) > 20.0


def _nice_step_125(rough_step: float) -> float:
    if not math.isfinite(rough_step) or rough_step <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(rough_step))
    base = 10 ** exponent
    for multiplier in (1, 2, 5, 10):
        step = multiplier * base
        if step >= rough_step:
            return float(step)
    return float(10 * base)


def _support_range(series: Sequence[BinnedSeries]) -> tuple[float, float]:
    mids_with_median = [
        mid
        for item in series
        for mid, median in zip(item.mids, item.q50)
        if median is not None
    ]
    if mids_with_median:
        lower = min(mids_with_median)
        upper = max(mids_with_median)
        if upper <= lower:
            upper = lower + 1.0
        return lower, upper

    mids = [mid for item in series for mid in item.mids]
    if not mids:
        return 0.0, 1.0
    lower = min(mids)
    upper = max(mids)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def apply_axis_style(ax, series: Sequence[BinnedSeries], x_label: str, y_label: str) -> None:
    log_y = y_is_log_worthy(series)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel(f"{y_label} (log scale)")
    else:
        ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    x_min, x_max = _support_range(series)
    span = max(x_max - x_min, 1e-9)
    major_step = _nice_step_125(span / 10.0)
    if x_min >= 0.0:
        x_start = 0.0
    else:
        x_start = major_step * math.floor(x_min / major_step)
    base = major_step * math.floor(x_max / major_step)
    half_step = major_step / 2.0
    x_stop = base + half_step if x_max <= base + half_step else base + major_step
    if x_stop <= x_start:
        x_stop = x_start + major_step

    from matplotlib import ticker as mticker

    ax.set_xlim(x_start, x_stop)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(major_step / 5.0))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)


def _family_variant(index: int) -> tuple[str, object]:
    return (
        FAMILY_PALETTE[index % len(FAMILY_PALETTE)],
        FAMILY_LINESTYLES[index % len(FAMILY_LINESTYLES)],
    )


def _draw_median_line(ax, series: BinnedSeries, *, color: str, linestyle, linewidth: float = MEDIAN_LINEWIDTH) -> None:
    median_x = [mid for mid, value in zip(series.mids, series.q50) if value is not None]
    median_y = [value for value in series.q50 if value is not None]
    if median_x:
        ax.plot(median_x, median_y, color=color, linewidth=linewidth, linestyle=linestyle, zorder=3)


def _draw_band(ax, series: BinnedSeries, color: str, median_linestyle) -> None:
    g95_x = [mid for mid, lo, hi in zip(series.mids, series.q05, series.q95) if lo is not None and hi is not None]
    g95_lo = [lo for lo, hi in zip(series.q05, series.q95) if lo is not None and hi is not None]
    g95_hi = [hi for lo, hi in zip(series.q05, series.q95) if lo is not None and hi is not None]
    if g95_x:
        ax.fill_between(g95_x, g95_lo, g95_hi, facecolor=color, alpha=OUTER_BAND_ALPHA, edgecolor="none")

    giqr_x = [mid for mid, lo, hi in zip(series.mids, series.q25, series.q75) if lo is not None and hi is not None]
    giqr_lo = [lo for lo, hi in zip(series.q25, series.q75) if lo is not None and hi is not None]
    giqr_hi = [hi for lo, hi in zip(series.q25, series.q75) if lo is not None and hi is not None]
    if giqr_x:
        ax.fill_between(giqr_x, giqr_lo, giqr_hi, facecolor=color, alpha=INNER_BAND_ALPHA, edgecolor="none")

    _draw_median_line(ax, series, color="black", linestyle=median_linestyle)


def _draw_family_line(ax, series: BinnedSeries, *, color: str, linestyle) -> None:
    _draw_median_line(ax, series, color=color, linestyle=linestyle)


def render_status_plot(plt, counts: dict[str, int], title: str, output_path: str, status_order: Sequence[str]) -> None:
    keys = [key for key in status_order if key in counts]
    values = [counts[key] for key in keys]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(keys, values, color=["#6C757D", "#D55E00", "#009E73", "#9E9E9E"][: len(keys)])
    for bar, value in zip(bars, values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value}\n{pct:.1f}%", ha="center", va="bottom")
    ax.set_title(title)
    ax.set_xlabel("status")
    ax.set_ylabel("rows")
    ax.grid(True, axis="y", alpha=0.25)
    _finalize_figure(fig)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_discrepancy_failure_counts_plot(plt, discrepancy_info: dict[str, dict[str, object]], title: str, output_path: str) -> None:
    solvers = list(discrepancy_info.keys())
    fig, ax = plt.subplots(figsize=(max(8.0, 2.8 * len(solvers)), 5.5))
    xs = list(range(len(solvers)))
    valid = [int(discrepancy_info[solver]["valid"]) for solver in solvers]
    invalid = [int(discrepancy_info[solver]["invalid"]) for solver in solvers]
    missing = [int(discrepancy_info[solver]["missing"]) for solver in solvers]

    ax.bar(xs, valid, color="#009E73", label="valid")
    ax.bar(xs, invalid, bottom=valid, color="#D55E00", label="invalid")
    stacked = [v + i for v, i in zip(valid, invalid)]
    ax.bar(xs, missing, bottom=stacked, color="#B0B0B0", label="missing")

    for index, solver in enumerate(solvers):
        total = int(discrepancy_info[solver]["rows_total"])
        failure_rate = float(discrepancy_info[solver]["failure_rate"]) * 100.0
        ax.text(xs[index], stacked[index] + missing[index], f"{invalid[index]}/{total}\n{failure_rate:.1f}%", ha="center", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels(solvers, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("rows")
    ax.grid(True, axis="y", alpha=0.25)
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    _finalize_figure(fig, top=0.94, right=0.98)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_discrepancy_failure_reasons_plot(plt, discrepancy_info: dict[str, dict[str, object]], title: str, output_path: str) -> None:
    all_reasons = sorted({reason for payload in discrepancy_info.values() for reason in payload.get("reasons", {}).keys()})
    if not all_reasons:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No discrepancy timing invalidations found.", ha="center", va="center")
        _finalize_figure(fig)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return

    solvers = list(discrepancy_info.keys())
    fig, ax = plt.subplots(figsize=(max(8.5, 3.0 * len(solvers)), 6.0))
    xs = list(range(len(solvers)))
    bottoms = [0] * len(solvers)
    palette = list(FAMILY_PALETTE)
    while len(palette) < len(all_reasons):
        palette.extend(FAMILY_PALETTE)

    for reason, color in zip(all_reasons, palette):
        heights = [int(discrepancy_info[solver].get("reasons", {}).get(reason, 0)) for solver in solvers]
        ax.bar(xs, heights, bottom=bottoms, label=reason, color=color)
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_xticks(xs)
    ax.set_xticklabels(solvers, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("invalid rows")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _finalize_figure(fig, right=0.8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_empty_panel(ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")


def render_ac_modeling_cost_overview_plot(
    plt,
    overview: dict[str, dict[str, object]],
    title: str,
    output_path: str,
) -> None:
    dummy_payload = overview.get("ac_fft_dummy_work_share_avg", {})
    dummy_values = [
        float(value)
        for value in dummy_payload.get("values", [])
        if value is not None and math.isfinite(float(value))
    ]

    show_dummy_panel = bool(dummy_values)
    if show_dummy_panel:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        ax_left = axes[0]
    else:
        fig, ax_left = plt.subplots(1, 1, figsize=(8.5, 5.0))

    solver_names = [
        solver_name
        for solver_name, payload in overview.items()
        if solver_name != "ac_fft_dummy_work_share_avg"
        and isinstance(payload, dict)
        and any(
            payload.get(key) is not None
            for key in ("small_share_median", "big_share_median", "slack_share_median")
        )
    ]
    if not solver_names:
        solver_names = ["ac_batch", "ac_fft"]
    xs = list(range(len(solver_names)))
    small = [overview.get(solver, {}).get("small_share_median") or 0.0 for solver in solver_names]
    big = [overview.get(solver, {}).get("big_share_median") or 0.0 for solver in solver_names]
    slack = [overview.get(solver, {}).get("slack_share_median") or 0.0 for solver in solver_names]

    ax_left.bar(xs, small, color="#56B4E9", label="small")
    ax_left.bar(xs, big, bottom=small, color="#E69F00", label="big")
    stacked = [left + right for left, right in zip(small, big)]
    ax_left.bar(xs, slack, bottom=stacked, color="#D55E00", label="slack")
    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(solver_names)
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("median runtime share")
    ax_left.set_title("AC base runtime share by block kind")
    ax_left.grid(True, axis="y", alpha=0.25)

    if show_dummy_panel:
        ax_right = axes[1]
        bins = min(20, max(6, int(math.sqrt(len(dummy_values)))))
        ax_right.hist(dummy_values, bins=bins, color="#0072B2", alpha=0.8)
        ax_right.set_xlim(0.0, 1.0)
        ax_right.set_xlabel("dummy-work share")
        ax_right.set_ylabel("rows")
        ax_right.set_title("ac_fft dummy-work share")
        ax_right.grid(True, axis="y", alpha=0.25)

    fig.suptitle(title)
    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        legend_labels = [
            _share_legend_label("small", solver_names, small),
            _share_legend_label("big", solver_names, big),
            _share_legend_label("slack", solver_names, slack),
        ]
        fig.legend(handles, legend_labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.01))
    _finalize_figure(fig, top=0.88, right=0.98)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_acdc_useful_grid_overview_plot(
    plt,
    payload: dict[str, object],
    title: str,
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(10.5, 8.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.85])
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[1, 0])

    _render_runtime_vs_useful_scatter(
        ax_scatter,
        payload,
        "ac_discrepancy runtime vs useful grid share",
        "Greens",
        include_padding=False,
    )

    useful = _mean(payload.get("useful_grid_share_values", []))
    rejected = _mean(payload.get("rejected_grid_share_values", []))
    if useful is None and rejected is None:
        _render_empty_panel(ax_bar, "Mean capped-grid composition", "No usable ACDC grid-share data found.")
    else:
        _draw_component_barh(
            ax_bar,
            "ac_discrepancy",
            [
                ("useful grid share", useful or 0.0, "#009E73"),
                ("rejected grid share", rejected or 0.0, "#D55E00"),
            ],
        )
        ax_bar.set_xlim(0.0, 1.0)
        ax_bar.set_xlabel("share")
        ax_bar.set_title("Mean capped-grid composition")
        ax_bar.grid(True, axis="x", alpha=0.25)

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#009E73", label="useful grid share"),
            Patch(facecolor="#D55E00", label="rejected grid share"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(title)
    _finalize_figure(fig, top=0.9, right=0.98)
    fig.subplots_adjust(bottom=0.14, hspace=0.34)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_fft_utilization_overview_plot(
    plt,
    overview: dict[str, object],
    title: str,
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(13.0, 9.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_ac = fig.add_subplot(grid[0, 0])
    ax_discrepancy = fig.add_subplot(grid[0, 1])
    ax_composition = fig.add_subplot(grid[1, :])
    ac_payload = _safe_solver_payload(overview, "ac_fft")
    discrepancy_payload = _safe_solver_payload(overview, "discrepancy")

    _render_runtime_vs_useful_scatter(
        ax_ac,
        ac_payload,
        "ac_fft runtime vs used grid share",
        "Blues",
        include_padding=True,
        share_label="used grid share",
        stats_box_position="bottom_left",
    )
    _render_runtime_vs_useful_scatter(
        ax_discrepancy,
        discrepancy_payload,
        "discrepancy runtime vs used grid share",
        "Greens",
        include_padding=False,
        share_label="used grid share",
        stats_box_position="bottom_left",
    )

    useful_color = "#009E73"
    non_useful_color = "#E69F00"
    ax = ax_composition
    ac_useful = _mean(ac_payload.get("useful_grid_share_values", []))
    ac_empty = _mean(ac_payload.get("empty_grid_share_values", []))
    ac_padding = _mean(ac_payload.get("padding_waste_share_values", []))
    discrepancy_useful = _mean(discrepancy_payload.get("useful_grid_share_values", []))
    discrepancy_empty = _mean(discrepancy_payload.get("empty_grid_share_values", []))
    if ac_useful is None and discrepancy_useful is None:
        _render_empty_panel(ax, "Mean FFT grid composition", "No usable grid-composition data found.")
    else:
        _draw_component_barh(
            ax,
            "ac_fft",
            [
                ("used grid share", ac_useful or 0.0, useful_color),
                ("empty/padding share", (ac_empty or 0.0) + (ac_padding or 0.0), non_useful_color),
            ],
        )
        _draw_component_barh(
            ax,
            "discrepancy",
            [
                ("used grid share", discrepancy_useful or 0.0, useful_color),
                ("empty/padding share", discrepancy_empty or 0.0, non_useful_color),
            ],
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("share")
        ax.set_title("Mean FFT grid composition")
        ax.grid(True, axis="x", alpha=0.25)

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=useful_color, label="used grid share"),
            Patch(facecolor=non_useful_color, label="empty/padding share"),
        ],
        loc="lower left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.07, 0.035),
    )
    fig.suptitle(title)
    _finalize_figure(fig, top=0.9, right=0.98)
    fig.subplots_adjust(bottom=0.12, hspace=0.28)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_solver_small_multiples(
    plt,
    series_by_solver: dict[str, BinnedSeries],
    x_label: str,
    title: str,
    output_path: str,
    mat_update_color: str,
    other_color: str,
) -> None:
    if not series_by_solver:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No usable runtime data found.", ha="center", va="center")
        _finalize_figure(fig)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return

    items = list(series_by_solver.items())
    cols = 3 if len(items) > 4 else 2
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, 4.0 * rows), squeeze=False)
    all_series = [item for _, item in items]

    for ax, (time_col, series) in zip(axes.flatten(), items):
        color = mat_update_color if pretty_solver_name(time_col) == "mat_update" else other_color
        _draw_band(ax, series, color=color, median_linestyle="solid")
        ax.set_title(series.solver)
        apply_axis_style(ax, all_series, x_label, "runtime (ms)")

    for ax in axes.flatten()[len(items) :]:
        ax.axis("off")

    fig.suptitle(title)
    _finalize_figure(fig, top=0.95)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_solver_point_clouds(
    plt,
    points_by_solver: dict[str, list[tuple[float, float]]],
    x_label: str,
    title: str,
    output_path: str,
) -> None:
    items = [(time_col, points) for time_col, points in points_by_solver.items() if points]
    if not items:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No usable runtime data found.", ha="center", va="center")
        _finalize_figure(fig)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return

    all_x = [x for _, points in items for x, _ in points]
    all_y = [y for _, points in items for _, y in points]
    x_log = _positive_values_log_worthy(all_x)
    y_log = _values_log_worthy(all_y)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for index, (time_col, points) in enumerate(items):
        color, _ = _family_variant(index)
        ax.scatter(
            [x for x, _ in points],
            [y for _, y in points],
            s=18,
            color=color,
            alpha=0.50,
            linewidths=0.0,
            label=pretty_solver_name(time_col),
            rasterized=len(points) > 500,
        )

    if x_log:
        ax.set_xscale("log")
        ax.set_xlabel(f"{x_label} (log scale)")
    else:
        ax.set_xlabel(x_label)
    ax.set_xlim(*_x_limits_with_padding(all_x, log_scale=x_log))

    if y_log:
        ax.set_yscale("log")
        ax.set_ylabel("runtime (ms, log scale)")
    else:
        ax.set_ylabel("runtime (ms)")
    ax.set_ylim(*_runtime_y_limits_with_padding(all_y, log_scale=y_log))

    ax.set_title(title)
    ax.set_facecolor("#FBFBFB")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    _finalize_figure(fig)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_branch_bands(plt, family: str, series_by_solver: dict[str, BinnedSeries], x_label: str, title: str, output_path: str) -> None:
    items = list(series_by_solver.items())
    if not items:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, f"No usable runtime data found for {family}.", ha="center", va="center")
        _finalize_figure(fig)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    series_list = [item for _, item in items]

    for index, (_, series) in enumerate(items):
        color, linestyle = _family_variant(index)
        _draw_band(ax, series, color=color, median_linestyle=linestyle)

    apply_axis_style(ax, series_list, x_label, "runtime (ms)")
    ax.set_title(title)

    from matplotlib.lines import Line2D

    legend_handles = []
    for index, (_, series) in enumerate(items):
        color, linestyle = _family_variant(index)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=MEDIAN_LINEWIDTH, linestyle=linestyle, label=series.solver))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _finalize_figure(fig, right=0.8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_branch_bands_detail(
    plt,
    family: str,
    series_by_solver: dict[str, BinnedSeries],
    points_by_solver: dict[str, list[tuple[float, float]]],
    x_label: str,
    title: str,
    output_path: str,
) -> None:
    items = list(series_by_solver.items())
    if not items:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, f"No usable runtime data found for {family}.", ha="center", va="center")
        _finalize_figure(fig)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return

    detail_rows = len(items)
    fig = plt.figure(figsize=(18.0, max(6.4, 2.45 * detail_rows)))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.16)
    ax_overview = fig.add_subplot(grid[0, 0])
    detail_grid = grid[0, 1].subgridspec(detail_rows, 1, hspace=0.2)
    series_list = [item for _, item in items]

    for index, (_, series) in enumerate(items):
        color, linestyle = _family_variant(index)
        _draw_family_line(ax_overview, series, color=color, linestyle=linestyle)

    apply_axis_style(ax_overview, series_list, x_label, "runtime (ms)")
    ax_overview.set_title(f"{family}: family comparison")

    from matplotlib.lines import Line2D

    legend_handles = []
    for index, (_, series) in enumerate(items):
        color, linestyle = _family_variant(index)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=MEDIAN_LINEWIDTH, linestyle=linestyle, label=series.solver))
    ax_overview.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.95,
    )

    detail_axes = []
    shared_ax = None
    for index, (time_col, series) in enumerate(items):
        color, linestyle = _family_variant(index)
        ax = fig.add_subplot(detail_grid[index, 0], sharex=shared_ax, sharey=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        detail_axes.append(ax)

        points = points_by_solver.get(time_col, [])
        if points:
            ax.scatter(
                [x for x, _ in points],
                [y for _, y in points],
                s=12,
                color=color,
                alpha=0.9,
                linewidths=0.0,
                zorder=1,
                rasterized=len(points) > 500,
            )
        _draw_median_line(ax, series, color="black", linestyle=linestyle)
        apply_axis_style(ax, series_list, x_label, "runtime (ms)")
        ax.set_title(series.solver, loc="left", fontsize=10)
        if index != len(items) - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if index != len(items) // 2:
            ax.set_ylabel("")

    if detail_axes:
        detail_axes[0].text(
            0.98,
            1.08,
            "Per-solver raw points + median",
            transform=detail_axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(title)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.9, bottom=0.08, wspace=0.16)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
