from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


DEFAULT_NOCA_ALPHA6_Y = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_NOCA_ALPHA16_Y = [1]
DEFAULT_CONVERGENCE_PAIRS = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (16, 1)]
DEFAULT_INCLUDE_CFD_NOCA = True
DEFAULT_INCLUDE_CFD_CONVERGENCE = True
DEFAULT_SUPER_FAST = {
    # Keep full alpha/Y coverage, but use coarse/fast settings.
    "noca_alpha6_y": [1, 2, 3, 4, 5, 6],
    # "noca_alpha6_y": [1, 2, 3, 4],
    "noca_alpha16_y": [1],
    "noca_n_points": 6,
    # "conv_pairs": [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (16, 1)],
    "conv_pairs": [(6, 2)],
    "conv_parameter_names": ["iP", "dLx", "dLy"],
    # "conv_parameter_names": ["dLx"],
    "conv_data_types": ["CFD", "PIV"],
    "conv_fast_factor": 2.0,  # higher than 50 is not good
    "conv_is_small_piv": False,
    "conv_piv_sweep_n_points": 5,
    "include_cfd_noca": False,
    "include_cfd_convergence": False,
}
"""
 python -m kite_piv_analysis._process_calculating_noca_and_kutta__convergence_study --super-fast --skip-convergence
"""

"""
python -m kite_piv_analysis._process_calculating_noca_and_kutta__convergence_study \
  --skip-noca \
  --conv-pairs 6:4 \
  --conv-parameter-names iP,dLx,dLy \
  --conv-data-types CFD,PIV \
  --conv-fast-factor 1 \
  --conv-piv-sweep-n-points 6

"""


def _resolve_base_package() -> str:
    """
    Support both:
    1) python -m kite_piv_analysis._process_calculating_noca_and_kutta__convergence_study
    2) python src/kite_piv_analysis/_process_calculating_noca_and_kutta__convergence_study.py
    """
    if __package__:
        return __package__

    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return "kite_piv_analysis"


BASE_PACKAGE = _resolve_base_package()


def _import_package_module(module_name: str):
    return importlib.import_module(f"{BASE_PACKAGE}.{module_name}")


def _parse_int_list(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_pairs(raw: str) -> list[tuple[int, int]]:
    """
    Parse format like: "6:1,6:2,16:1"
    """
    raw = raw.strip()
    if not raw:
        return []

    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid pair '{token}'. Expected format '<alpha>:<y_num>', e.g. '6:1'."
            )
        alpha_str, y_str = token.split(":", 1)
        out.append((int(alpha_str.strip()), int(y_str.strip())))
    return out


def _pairs_to_cli(pairs: list[tuple[int, int]]) -> str:
    return ",".join(f"{alpha}:{y_num}" for alpha, y_num in pairs)


def run_calculating_noca_and_kutta(
    alpha6_y_nums: list[int],
    alpha16_y_nums: list[int],
    n_points: int,
    include_cfd: bool,
) -> None:
    calculating_noca_and_kutta = _import_package_module("calculating_noca_and_kutta")

    print("\n[1/2] Running calculating_noca_and_kutta ...")
    if alpha6_y_nums:
        print(
            f"  - alpha=6, y_nums={alpha6_y_nums}, n_points={n_points}, "
            f"include_cfd={include_cfd}"
        )
        calculating_noca_and_kutta.save_results_single_alpha(
            alpha=6,
            y_num_list=alpha6_y_nums,
            n_points=n_points,
            include_cfd=include_cfd,
            include_piv=True,
            reuse_existing_when_skipped=True,
        )

    if alpha16_y_nums:
        print(
            f"  - alpha=16, y_nums={alpha16_y_nums}, n_points={n_points}, "
            f"include_cfd={include_cfd}"
        )
        calculating_noca_and_kutta.save_results_single_alpha(
            alpha=16,
            y_num_list=alpha16_y_nums,
            n_points=n_points,
            include_cfd=include_cfd,
            include_piv=True,
            reuse_existing_when_skipped=True,
        )


def run_convergence_study(
    pairs: list[tuple[int, int]],
    parameter_names: list[str],
    data_types: list[str],
    is_small_piv: bool,
    fast_factor: float,
    piv_sweep_n_points: int,
    include_cfd: bool,
) -> None:
    convergence_study = _import_package_module("convergence_study")

    print("\n[2/2] Running convergence_study ...")
    effective_data_types = list(data_types)
    if not include_cfd:
        effective_data_types = [d for d in effective_data_types if d.upper() != "CFD"]
    if not effective_data_types:
        effective_data_types = ["PIV"]

    for alpha, y_num in pairs:
        print(
            f"  - alpha={alpha}, y_num={y_num}, params={parameter_names}, "
            f"data_types={effective_data_types}, fast_factor={fast_factor}, "
            f"is_small_piv={is_small_piv}, piv_sweep_n_points={piv_sweep_n_points}"
        )
        convergence_study.storing_and_collecting_results(
            alpha=alpha,
            y_num=y_num,
            parameter_names=parameter_names,
            data_types=effective_data_types,
            is_small_piv=is_small_piv,
            fast_factor=fast_factor,
        )
        convergence_study.storing_PIV_percentage_sweep(
            alpha=alpha,
            y_num=y_num,
            n_points=piv_sweep_n_points,
            data_type="PIV",
            fast_factor=fast_factor,
        )
        if any(d.upper() == "CFD" for d in effective_data_types):
            convergence_study.storing_PIV_percentage_sweep(
                alpha=alpha,
                y_num=y_num,
                n_points=piv_sweep_n_points,
                data_type="CFD",
                fast_factor=fast_factor,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run calculating_noca_and_kutta and convergence_study sequentially."
        )
    )
    parser.add_argument(
        "--skip-noca",
        action="store_true",
        help="Skip calculating_noca_and_kutta stage.",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip convergence_study stage.",
    )
    parser.add_argument(
        "--super-fast",
        action="store_true",
        help="Use DEFAULT_SUPER_FAST preset (coarse but full alpha/Y coverage).",
    )
    parser.add_argument(
        "--include-cfd-noca",
        action="store_true",
        help=(
            "Include CFD recomputation in calculating_noca_and_kutta. "
            f"Default is {DEFAULT_INCLUDE_CFD_NOCA}."
        ),
    )
    parser.add_argument(
        "--include-cfd-convergence",
        action="store_true",
        help=(
            "Include CFD in convergence_study data_types. "
            f"Default is {DEFAULT_INCLUDE_CFD_CONVERGENCE}."
        ),
    )

    parser.add_argument(
        "--noca-n-points",
        type=int,
        default=10,
        help="n_points for calculating_noca_and_kutta boundary sweeps.",
    )
    parser.add_argument(
        "--noca-alpha6-y",
        type=str,
        default="1,2,3,4,5,6,7",
        help="Comma-separated y_num list for alpha=6.",
    )
    parser.add_argument(
        "--noca-alpha16-y",
        type=str,
        default="1",
        help="Comma-separated y_num list for alpha=16.",
    )

    parser.add_argument(
        "--conv-pairs",
        type=str,
        default=_pairs_to_cli(DEFAULT_CONVERGENCE_PAIRS),
        help="Comma-separated alpha:y_num pairs, e.g. '6:1,6:2,16:1'.",
    )
    parser.add_argument(
        "--conv-parameter-names",
        type=str,
        default="iP,dLx,dLy",
        help="Comma-separated parameter names for convergence sweep.",
    )
    parser.add_argument(
        "--conv-data-types",
        type=str,
        default="CFD,PIV",
        help="Comma-separated data types for convergence sweep.",
    )
    parser.add_argument(
        "--conv-fast-factor",
        type=float,
        default=10.0,
        help="fast_factor for convergence study (higher is faster/coarser).",
    )
    parser.add_argument(
        "--conv-is-small-piv",
        action="store_true",
        help="Use compact PIV sweep window in convergence_study.",
    )
    parser.add_argument(
        "--conv-piv-sweep-n-points",
        type=int,
        default=10,
        help="n_points for storing_PIV_percentage_sweep.",
    )
    args = parser.parse_args()

    alpha6_y_nums = _parse_int_list(args.noca_alpha6_y) or DEFAULT_NOCA_ALPHA6_Y
    alpha16_y_nums = _parse_int_list(args.noca_alpha16_y) or DEFAULT_NOCA_ALPHA16_Y
    noca_n_points = args.noca_n_points

    conv_pairs = _parse_pairs(args.conv_pairs) or DEFAULT_CONVERGENCE_PAIRS
    conv_parameter_names = [
        s.strip() for s in args.conv_parameter_names.split(",") if s.strip()
    ] or ["iP", "dLx", "dLy"]
    conv_data_types = [
        s.strip() for s in args.conv_data_types.split(",") if s.strip()
    ] or [
        "CFD",
        "PIV",
    ]
    conv_fast_factor = args.conv_fast_factor
    conv_is_small_piv = args.conv_is_small_piv
    conv_piv_sweep_n_points = args.conv_piv_sweep_n_points
    include_cfd_noca = DEFAULT_INCLUDE_CFD_NOCA or args.include_cfd_noca
    include_cfd_convergence = (
        DEFAULT_INCLUDE_CFD_CONVERGENCE or args.include_cfd_convergence
    )

    if args.super_fast:
        alpha6_y_nums = list(DEFAULT_SUPER_FAST["noca_alpha6_y"])
        alpha16_y_nums = list(DEFAULT_SUPER_FAST["noca_alpha16_y"])
        noca_n_points = int(DEFAULT_SUPER_FAST["noca_n_points"])
        conv_pairs = list(DEFAULT_SUPER_FAST["conv_pairs"])
        conv_parameter_names = list(DEFAULT_SUPER_FAST["conv_parameter_names"])
        conv_data_types = list(DEFAULT_SUPER_FAST["conv_data_types"])
        conv_fast_factor = float(DEFAULT_SUPER_FAST["conv_fast_factor"])
        conv_is_small_piv = bool(DEFAULT_SUPER_FAST["conv_is_small_piv"])
        conv_piv_sweep_n_points = int(DEFAULT_SUPER_FAST["conv_piv_sweep_n_points"])
        # Preset defaults, but explicit CLI include flags always take precedence.
        include_cfd_noca = bool(DEFAULT_SUPER_FAST["include_cfd_noca"]) or bool(
            args.include_cfd_noca
        )
        include_cfd_convergence = bool(
            DEFAULT_SUPER_FAST["include_cfd_convergence"]
        ) or bool(args.include_cfd_convergence)
        print("Using DEFAULT_SUPER_FAST preset.")

    if not args.skip_noca:
        run_calculating_noca_and_kutta(
            alpha6_y_nums=alpha6_y_nums,
            alpha16_y_nums=alpha16_y_nums,
            n_points=noca_n_points,
            include_cfd=include_cfd_noca,
        )
    else:
        print("[1/2] Skipped calculating_noca_and_kutta.")

    if not args.skip_convergence:
        run_convergence_study(
            pairs=conv_pairs,
            parameter_names=conv_parameter_names,
            data_types=conv_data_types,
            is_small_piv=conv_is_small_piv,
            fast_factor=conv_fast_factor,
            piv_sweep_n_points=conv_piv_sweep_n_points,
            include_cfd=include_cfd_convergence,
        )
    else:
        print("[2/2] Skipped convergence_study.")

    print("\nDone.")


if __name__ == "__main__":
    main()
