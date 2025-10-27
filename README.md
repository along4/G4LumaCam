# G4LumaCam
A Geant4-based Simulator for LumaCam Event Camera

![screenshot](notebooks/lumacam_simulation.png)

## Overview
G4LumaCam is a simulation package for the LumaCam event camera that allows reconstruction of neutron events using the same analysis workflow as actual experimental data. This tool enables researchers to simulate, validate, and optimize detection setups before conducting physical experiments.

## Features
- High-fidelity simulation of neutron interactions based on Geant4 physics models
- Realistic optical ray tracing through the LumaCam lens system
- Seamless integration with the EMPIR reconstruction workflow
- Configurable neutron source properties (energy, spatial distribution, etc.)
- Multi-process support for efficient data processing

## Getting Started
Check out our [detailed tutorial](notebooks/tutorial.ipynb) for a step-by-step guide on using G4LumaCam, from simulation setup to data analysis.

## Dependencies
G4LumaCam relies on the following software packages:
- [Geant4 10.6](https://geant4.web.cern.ch/) - Particle physics simulation toolkit
- [ray-optics](https://github.com/mjhoptics/ray-optics) - Optical ray tracing library
- [lmfit](https://lmfit.github.io/lmfit-py/) - Non-Linear Least-Square Minimization and Curve-Fitting

Additionally, G4LumaCam depends on EMPIR, which is a proprietary, non-open source code used to reconstruct events recorded using the LumaCam Timepix-3 based detector. EMPIR can be purchased on request from [LoskoVision Ltd.](https://amscins.com/product/chronos-series/neutron-imaging/)

## Installation
### Prerequisites
It is recommended to install Geant4 through Docker:
```bash
docker pull jeffersonlab/geant4:g4v10.6.2-ubuntu24
```

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/TsvikiHirsh/G4LumaCam.git
   ```

2. (Optional) Specify the EMPIR executables path before installation:
   ```bash
   export EMPIR_PATH=/path/to/empir/executables
   ```

3. Navigate to the cloned directory and install:
   ```bash
   cd G4LumaCam
   pip install .
   ```

## EMPIR Configuration
G4LumaCam provides three ways to specify the path to EMPIR executables:

1. **Environment Variable (during installation)**: Set `EMPIR_PATH` before installation to configure it globally:
   ```bash
   export EMPIR_PATH=/path/to/empir/executables
   pip install .
   ```

2. **Runtime Parameter**: Specify the path when creating an Analysis object:
   ```python
   analysis = lumacam.Analysis(archive="your_archive", empir_dirpath="/path/to/empir/executables")
   ```

3. **Default Path**: If no path is specified, G4LumaCam will look for EMPIR in the `./empir` directory relative to your working directory.

## Usage Example
The following example demonstrates a typical workflow with G4LumaCam:
```python
# Run the simulation of the neutron source
import lumacam
sim = lumacam.Simulate("openbeam")
config = lumacam.Config.neutrons_uniform_energy()
df = sim.run(config)

# Trace rays through the lens
lens = lumacam.Lens(archive="openbeam")
lens.trace_rays(blob=1., deadtime=600) # define a blob radius size of 1 px and deadtime of 600 ns

# Analyse and process data using EMPIR
# Uses EMPIR path from installation config or default to run the standard EMPIR workflow to produce a tiff stack of the TOF-dependent images
analyse = lumacam.Analysis(archive="archive/test/openbeam")
analyse.process(params="hitmap",event2image=True)
```

## Citation
If you use G4LumaCam in your research, please cite:
```
@software{g4lumacam,
 author = {Hirsh, Tsviki Y.},
 title = {G4LumaCam: A Geant4-based Simulator for LumaCam Event Camera},
 url = {https://github.com/TsvikiHirsh/G4LumaCam},
 year = {2025},
}
```

## License
G4LumaCam is released under the MIT License for original code contributions, see [LICENSE](LICENSE.md)