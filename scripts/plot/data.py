from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass
class Row:
    raw: Dict[str, str]
    values: Dict[str, Optional[float]]
    vectors: Dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockRuntimeBreakdown:
    values: list[float]
    total: float
    small_total: float
    big_total: float
    slack_total: float

    @property
    def small_share(self) -> float:
        return self.small_total / self.total

    @property
    def big_share(self) -> float:
        return self.big_total / self.total

    @property
    def slack_share(self) -> float:
        return self.slack_total / self.total


@dataclass(frozen=True)
class TimingStatusColumns:
    valid: Optional[str] = None
    fail: Optional[str] = None
    reason: Optional[str] = None


DEFAULT_SOLVER_ORDER = (
    "mat_update",
    "gur",
    "ac_legacy",
    "ac_naive",
    "ac_batch",
    "ac_discrepancy",
    "ac_fft",
    "gupta",
    "gupta_batch",
    "discrepancy",
)
_SOLVER_ORDER_INDEX = {name: index for index, name in enumerate(DEFAULT_SOLVER_ORDER)}


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def parse_float_vector(value: str) -> list[float]:
    text = (value or "").strip()
    if not text:
        return []
    out: list[float] = []
    for part in text.split(";"):
        parsed = parse_float(part)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def read_csv_rows(path: str) -> tuple[list[str], list[Row]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        rows: list[Row] = []
        for raw_row in reader:
            raw_values = {name: (raw_row.get(name, "") or "").strip() for name in fieldnames}
            values = {name: parse_float(raw_values.get(name, "")) for name in fieldnames}
            vectors = {
                name: parse_float_vector(raw_value)
                for name, raw_value in raw_values.items()
                if ";" in raw_value
            }
            rows.append(Row(raw=raw_values, values=values, vectors=vectors))
    return fieldnames, rows


def _row_int_value(row: Row, key: str) -> Optional[int]:
    value = row.values.get(key)
    if value is None or not math.isfinite(value):
        return None
    return int(value)


def block_runtime_breakdown(row: Row, vector_column: str) -> Optional[BlockRuntimeBreakdown]:
    values = row.vectors.get(vector_column, [])
    if not values:
        return None

    num_small = _row_int_value(row, "num_small_blocks")
    num_big = _row_int_value(row, "num_big_blocks")
    if num_small is None or num_big is None:
        return None

    base_count = num_small + num_big
    if len(values) not in {base_count, base_count + 1}:
        return None

    small_total = sum(values[:num_small])
    big_total = sum(values[num_small : num_small + num_big])
    slack_total = sum(values[num_small + num_big :])
    total = small_total + big_total + slack_total
    if not math.isfinite(total) or total <= 0.0:
        return None

    return BlockRuntimeBreakdown(
        values=list(values),
        total=total,
        small_total=small_total,
        big_total=big_total,
        slack_total=slack_total,
    )


def is_time_column(name: str) -> bool:
    return name.lower().startswith(("ns_", "us_", "ms_", "s_"))


def unit_scale_to_ms(name: str) -> float:
    lname = name.lower()
    if lname.startswith("ns_"):
        return 1.0 / 1_000_000.0
    if lname.startswith("us_"):
        return 1.0 / 1_000.0
    if lname.startswith("ms_"):
        return 1.0
    if lname.startswith("s_"):
        return 1_000.0
    return 1.0


def pretty_solver_name(col: str) -> str:
    for prefix in ("ns_", "us_", "ms_", "s_"):
        if col.lower().startswith(prefix):
            return col[len(prefix) :]
    return col


def is_excluded_series(col: str) -> bool:
    solver = pretty_solver_name(col).lower()
    return solver in {"min", "max"} or solver.endswith("_base_block_totals")


def solver_sort_key(col: str) -> tuple[int, str]:
    solver = pretty_solver_name(col)
    return (_SOLVER_ORDER_INDEX.get(solver, 10_000), solver)


def ordered_time_columns(fieldnames: Sequence[str]) -> list[str]:
    cols = [name for name in fieldnames if is_time_column(name) and not is_excluded_series(name)]
    return sorted(cols, key=solver_sort_key)


def normalize_reason(reason: str) -> str:
    text = (reason or "").strip()
    return text if text else "<unknown>"


def timing_status_columns_for_time_col(
    name: str, fieldnames: Sequence[str]
) -> Optional[TimingStatusColumns]:
    label = pretty_solver_name(name)
    header = set(fieldnames)

    if label == "discrepancy":
        if "valid_discrepancy" in header:
            return TimingStatusColumns(
                valid="valid_discrepancy",
                reason="fallback_reason_discrepancy",
            )
    if label == "ac_discrepancy":
        if "valid_ac_discrepancy" in header:
            return TimingStatusColumns(
                valid="valid_ac_discrepancy",
                reason="fallback_reason_ac_discrepancy",
            )
    return None


def row_has_invalid_timing(row: Row, status_cols: TimingStatusColumns) -> bool:
    if status_cols.valid:
        valid = row.values.get(status_cols.valid)
        if valid is not None:
            return int(valid) == 0
    if status_cols.fail:
        fail = row.values.get(status_cols.fail)
        if fail is not None:
            return int(fail) != 0
    return False


def timing_state(row: Row, time_col: str, fieldnames: Sequence[str]) -> str:
    raw_value = row.raw.get(time_col, "").strip()
    if not raw_value:
        return "missing"
    status_cols = timing_status_columns_for_time_col(time_col, fieldnames)
    if status_cols is not None and row_has_invalid_timing(row, status_cols):
        return "invalid"
    return "usable"


def solver_family_for_name(solver: str) -> str:
    if solver == "mat_update":
        return "mat_update"
    if solver == "gur":
        return "gur"
    if solver.startswith("ac_"):
        return "ac"
    if solver.startswith("gupta"):
        return "gupta"
    if solver.startswith("discrepancy"):
        return "discrepancy"
    return "other"


def solver_family_for_time_col(time_col: str) -> str:
    return solver_family_for_name(pretty_solver_name(time_col))


def infer_solver_families(time_cols: Sequence[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for col in sorted(time_cols, key=solver_sort_key):
        families[solver_family_for_time_col(col)].append(col)
    return dict(families)
