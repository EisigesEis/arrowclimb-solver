from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from .data import Row

MetricFn = Callable[[Row], Optional[float]]


def has_proxy_s_inputs(fieldnames: Sequence[str]) -> bool:
    available = set(fieldnames)
    base_required = {"d", "p_max", "Delta", "nmax", "mmax", "machine_types"}
    if not base_required.issubset(available):
        return False
    return {"ell", "calN"}.issubset(available)


def has_proxy_s_no_big_inputs(fieldnames: Sequence[str]) -> bool:
    available = set(fieldnames)
    base_required = {"d", "p_max", "Delta", "nmax", "mmax", "machine_types"}
    if not base_required.issubset(available):
        return False
    return {"ell", "calN"}.issubset(available) or "num_big_blocks" in available


def row_proxy_s(row: Row) -> Optional[float]:
    d = row.values.get("d")
    p_max = row.values.get("p_max")
    delta = row.values.get("Delta")
    nmax = row.values.get("nmax")
    mmax = row.values.get("mmax")
    ell = row.values.get("ell")
    cal_n = row.values.get("calN")
    machine_types = row.values.get("machine_types")
    if any(v is None for v in (d, p_max, delta, nmax, mmax, ell, cal_n, machine_types)):
        return None
    p_term = max(p_max, delta)
    rhs = max(nmax, mmax, ell, cal_n)
    return d * math.log1p(machine_types * d * (p_term + 1.0)) + math.log1p(math.log1p(rhs))


def row_proxy_s_no_big(row: Row) -> Optional[float]:
    d = row.values.get("d")
    p_max = row.values.get("p_max")
    delta = row.values.get("Delta")
    nmax = row.values.get("nmax")
    mmax = row.values.get("mmax")
    ell = row.values.get("ell")
    cal_n = row.values.get("calN")
    machine_types = row.values.get("machine_types")
    num_big_blocks = row.values.get("num_big_blocks")
    if any(v is None for v in (d, p_max, delta, nmax, mmax, machine_types)):
        return None
    p_term = max(p_max, delta)
    if num_big_blocks is not None and num_big_blocks <= 0:
        rhs = max(nmax, mmax)
    else:
        if ell is None or cal_n is None:
            return None
        rhs = max(nmax, mmax, ell, cal_n)
    return d * math.log1p(machine_types * d * (p_term + 1.0)) + math.log1p(math.log1p(rhs))


METRICS: dict[str, MetricFn] = {
    "proxy_S": row_proxy_s,
    "proxy_S_no_big": row_proxy_s_no_big,
}


def choose_x_axis(
    fieldnames: Sequence[str],
    rows: Sequence[Row],
    time_cols: Sequence[str],
    explicit: Optional[str],
) -> str:
    if explicit:
        if explicit in METRICS:
            if explicit == "proxy_S" and not has_proxy_s_inputs(fieldnames):
                raise ValueError(
                    "Requested x-axis 'proxy_S' needs d, p_max, Delta, nmax, mmax, ell, calN, and machine_types."
                )
            if explicit == "proxy_S_no_big" and not has_proxy_s_no_big_inputs(fieldnames):
                raise ValueError(
                    "Requested x-axis 'proxy_S_no_big' needs d, p_max, Delta, nmax, mmax, machine_types and either ell/calN or num_big_blocks."
                )
            return explicit
        if explicit not in fieldnames:
            raise ValueError(f"Requested x-axis '{explicit}' not found in CSV header")
        return explicit

    if has_proxy_s_inputs(fieldnames):
        return "proxy_S"

    time_set = set(time_cols)
    candidates: list[tuple[int, int, str]] = []
    for name in fieldnames:
        if name in time_set:
            continue
        lname = name.lower()
        if lname == "status":
            continue
        finite = [row.values.get(name) for row in rows if row.values.get(name) is not None]
        unique = {value for value in finite}
        if len(unique) < 2:
            continue

        score = 0
        if lname == "proxy_s":
            score += 1500
        if lname == "avg_makespan":
            score += 1300
        if lname in {"load_total", "capacity_total"}:
            score += 1150
        if lname in {"jobs_total", "machines_total", "p_max", "speed_ratio"}:
            score += 700
        if lname == "c_guess":
            score += 1000
        if "guess" in lname:
            score += 600
        if "size" in lname or lname in {"n", "out_n", "m", "jobs", "k"}:
            score += 200
        if lname == "name":
            score -= 500
        score += min(len(unique), 500)
        candidates.append((score, len(unique), name))

    if not candidates:
        return "__row_index__"
    candidates.sort(reverse=True)
    return candidates[0][2]


def x_value_for_row(row: Row, x_axis: str, row_index: int) -> Optional[float]:
    if x_axis == "__row_index__":
        return float(row_index)
    metric = METRICS.get(x_axis)
    if metric is not None:
        return metric(row)
    return row.values.get(x_axis)


def x_axis_label(x_axis: str) -> str:
    return "row index" if x_axis == "__row_index__" else x_axis
