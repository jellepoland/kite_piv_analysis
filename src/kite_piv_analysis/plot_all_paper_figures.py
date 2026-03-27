from __future__ import annotations

import importlib
import sys
from pathlib import Path


if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    BASE_PACKAGE = "kite_piv_analysis"
else:
    BASE_PACKAGE = __package__


def main() -> None:
    """
    Backward-compatible wrapper.
    For figures + tables, use _plot_all_figures_and_print_all_tables.py.
    """
    runner = importlib.import_module(
        f"{BASE_PACKAGE}._plot_all_figures_and_print_all_tables"
    )
    runner.main(run_figures=True, run_tables=False, smoke=False)


if __name__ == "__main__":
    main()
