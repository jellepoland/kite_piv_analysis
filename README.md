[![CODECHECK](https://codecheck.org.uk/img/codeworks-badge.svg)](https://zenodo.org/records/18890103)

# Flow Field Analysis of a Leading-Edge Inflatable Kite Rigid Scale Model Using Stereoscopic Particle Image Velocimetry
This repository contains code that generates the figures a paper, titled: ["Flow Field Analysis of a Leading-Edge Inflatable Kite Rigid Scale Model Using Stereoscopic Particle Image Velocimetry"](https://wes.copernicus.org/preprints/wes-2025-217/wes-2025-217.pdf) published open-access in Wind Energy Science.

## Usage instructions
1. Install the repository:
   Linux: 
    ```bash
    git clone git@github.com:jellepoland/kite_piv_analysis.git && \
    cd kite_piv_analysis && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install -e .[dev]
    ```
    Windows:
    ```bash
    git clone git@github.com:jellepoland/kite_piv_analysis.git; `
    cd kite_piv_analysis; `
    python -m venv venv; `
    .\venv\Scripts\Activate.ps1; `
    pip install -e .[dev]
    ```
2. Download the necessary data from [Zenodo](https://doi.org/10.5281/zenodo.17395913) and place inside the `data/` folder.

3. Make sure you have a LaTeX distribution installed (e.g. `texlive-full` on Linux, MacTeX on macOS, MiKTeX on Windows) for rendering figure labels. Alternatively, you may need to adjust the Matplotlib configuration in the code to disable LaTeX rendering.

4. To process the data and generate the figures, run the following from the repository root (this will likely take hours):
    ```bash
    python ./src/kite_piv_analysis/_main_process_and_plot.py
    ```

### Requirements

Automatically installed via `pip install -e .[dev]`:

- `numpy`
- `pandas>=1.5.3`
- `matplotlib>=3.7.1`
- `cycler`
- `scipy`
- `VSM @ git+https://github.com/awegroup/Vortex-Step-Method.git@v2.1.0`

Needs to be installed separately:
- **LaTeX**: A LaTeX distribution (e.g. `texlive-full` on Linux, MacTeX on macOS, MiKTeX on Windows) is required for rendering figure labels. Matplotlib is configured with `text.usetex: True`.
- **Data**: The data required to generate the figures is not included in this repository due to size constraints. It can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.17395913) and should be placed inside the `data/` folder.
- **Python**: Python 3.8 or higher is recommended for running the code.

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :warning: License and Waiver

Specify the license under which your software is distributed and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [2026] Jelle Poland
