from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

TOOL_VERSION = "plot-suite-v5"
DEFAULT_BINS = 20
DEFAULT_MIN_BIN_N = 10
DEFAULT_TITLE_PREFIX = "Solver benchmark"

STATUS_ORDER = ("-1", "0", "1", "missing")

SINGLE_SOLVER_BLUE = "#0072B2"
SINGLE_SOLVER_ORANGE = "#E69F00"
FAMILY_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#7B61FF",  # violet
    "#D81B60",  # magenta-red
    "#5D4037",  # brown
)
FAMILY_LINESTYLES = (
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (3, 1)),
)

OUTER_BAND_ALPHA = 0.12
INNER_BAND_ALPHA = 0.24
MEDIAN_LINEWIDTH = 2.0


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    out_dir: str
    mode: str
    x_axes: Optional[Tuple[str, ...]]
    only_success: bool
    title_prefix: str
    force: bool
    bins: int
    min_bin_n: int
    families: Optional[Tuple[str, ...]] = None
    out_dir_is_default: bool = False

    def normalized_options(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "x_axes": list(self.x_axes or ()),
            "only_success": self.only_success,
            "title_prefix": self.title_prefix,
            "bins": self.bins,
            "min_bin_n": self.min_bin_n,
            "families": list(self.families or ()),
        }


@dataclass(frozen=True)
class PlotSpec:
    key: str
    filename: str
    kind: str
    title: str
    x_axis_dependent: bool
    family: Optional[str] = None
    solvers: Tuple[str, ...] = ()

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "filename": self.filename,
            "kind": self.kind,
            "title": self.title,
            "x_axis_dependent": self.x_axis_dependent,
            "family": self.family,
            "solvers": list(self.solvers),
        }
