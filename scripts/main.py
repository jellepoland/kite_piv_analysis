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
    import plot_reflection_and_reinterpolation
    import plot_spanwise_CFD_slices_p_and_v
    import plot_spanwise_CFD_slices_w
    import print_quantitative_results
    import print_uncertainty
    import calculating_chordwise_slice_loads

    # import force_from_pressure

    # data/CFD --> processed_data/CFD
    transforming_paraview_output.main()  # works

    # 1. runs and stores all convergence studies
    # 2. plots, saves the results
    convergence_study.main()  # works, 60min

    # Plots bounds over CFD & PIV, adds interpolation windows
    plot_bounds_CFD_PIV.main()  # works

    # 1. runs VSM, and saves for alpha = 6
    # 2. runs circulation, NOCA for each case, for the 10x10grid, averages and saves
    # 3. plots gamma distribution of VSM and for the CFD,PIV shapes
    plot_gamma_distribution.main()  # works, 30min

    # plots PIV results with and without mask
    plot_normal_masked_interpolated.main()  # works

    # plots qualitative results of CFD and PIV
    plot_qualitative_CFD_PIV.main()

    # plots spanwise CFD showing u,v,w, Q_criterion and lamba_2
    plot_spanwise_CFD_slices_p_and_v.main()  # works

    # plots spanwise CFD slices of w, demonstrating abs(w) < 3
    plot_spanwise_CFD_slices_w.main()  # works

    # extracts uncertainty from PIV data
    print_uncertainty.main()  # works

    # calculates loads on the chordwise slices, using p-integration of surface
    # calculating_chordwise_slice_loads.main()  # works, but integrated in next function
    print_quantitative_results.main()

    ##TODO: NOT NEEDED?
    ## plot_reflection_and_reinterpolation.main() #works


if __name__ == "__main__":
    main()
