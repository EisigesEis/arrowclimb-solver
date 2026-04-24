from __future__ import annotations

import os
import unittest
import uuid

from scripts.plot.cache import build_output_fingerprint, should_skip_output
from scripts.plot.config import PlotSpec, RunConfig


class CacheTests(unittest.TestCase):
    def test_should_skip_output_requires_existing_file_and_matching_fingerprint(self) -> None:
        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)
        output_path = os.path.join(build_dir, f"plot_cache_test_{uuid.uuid4().hex}.png")
        try:
            with open(output_path, "wb") as handle:
                handle.write(b"data")

            config = RunConfig(
                input_path="instances/E1/main.csv",
                out_dir="build/plots",
                mode="main",
                x_axes=("proxy_S", "avg_makespan"),
                only_success=False,
                title_prefix="Solver benchmark",
                force=False,
                bins=20,
                min_bin_n=10,
                families=("ac", "gupta"),
            )
            plot_spec = PlotSpec(
                key="branch_ac_bands",
                filename="branch_ac_bands.png",
                kind="branch_bands",
                title="Solver benchmark: ac branch runtime bands",
                x_axis_dependent=True,
                family="ac",
                solvers=("ac_legacy", "ac_naive"),
            )
            fingerprint = build_output_fingerprint("input-fp", config, plot_spec)
            manifest = {
                "outputs": {
                    "branch_ac_bands": {
                        "fingerprint": fingerprint,
                    }
                }
            }

            self.assertTrue(should_skip_output(output_path, manifest, "branch_ac_bands", fingerprint, False))
            self.assertFalse(should_skip_output(output_path, manifest, "branch_ac_bands", "different", False))
            self.assertFalse(should_skip_output(output_path, manifest, "branch_ac_bands", fingerprint, True))

            os.remove(output_path)
            self.assertFalse(should_skip_output(output_path, manifest, "branch_ac_bands", fingerprint, False))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
