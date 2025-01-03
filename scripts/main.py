from pathlib import Path
from utils import project_dir
from plotting import PlotParams


def main():
    import transforming_paraview_output
    import convergence_study
    import plot_bounds_CFD_PIV
    import plot_gamma_distribution
    import plot_normal_masked_interpolated
    import plot_qualitative_CFD_PIV
    import scripts.irrelevant.plot_reflection_and_reinterpolation as plot_reflection_and_reinterpolation
    import plot_spanwise_CFD_slices_p_and_v
    import plot_spanwise_CFD_slices_w
    import print_quantitative_results
    import scripts.irrelevant.print_uncertainty as print_uncertainty
    import scripts.calculating_integrated_surface_pressure as calculating_integrated_surface_pressure
    import calculating_noca_and_kutta

    # import force_from_pressure

    # data/CFD --> processed_data/CFD
    transforming_paraview_output.main()  # works

    # extracts uncertainty from PIV data
    print_uncertainty.main()  # works

    # plots PIV results with and without mask
    plot_normal_masked_interpolated.main()  # works
    # plots spanwise CFD slices of w, demonstrating abs(w) < 3
    plot_spanwise_CFD_slices_w.main()  # works

    # plots qualitative results of CFD and PIV
    plot_qualitative_CFD_PIV.main()  # works

    # Plots bounds over CFD & PIV, adds interpolation windows
    plot_bounds_CFD_PIV.main()  # works
    # 1. runs and stores all convergence studies
    # 2. plots, saves the results
    convergence_study.main()  # works, 60min

    ### Quantitative results
    # runs circulation, NOCA for each case, for the 10x10grid, averages and saves
    calculating_noca_and_kutta.main()  # works, 30min
    # plots gamma distribution of VSM and for the CFD,PIV shapes
    plot_gamma_distribution.main()  # works (relies on above)
    # integrates surface pressure for each case, saves
    calculating_integrated_surface_pressure.main()  # works

    # plots spanwise CFD showing u,v,w, and lamba_2
    plot_spanwise_CFD_slices_p_and_v.main()  # works


if __name__ == "__main__":
    main()
