# G4LumaCam

A Geant4-based Simulator for LumaCam Event Camera

![screenshot](notebooks/G4LumaCam.png)

## Overview

G4LumaCam is a simulation package for the LumaCam event camera that allows reconstruction of neutron events using the same analysis workflow as actual experimental data. This tool enables researchers to simulate, validate, and optimize detection setups before conducting physical experiments.

## Features

- High-fidelity simulation of neutron interactions based on Geant4 physics models
- Realistic optical ray tracing through the LumaCam lens system
- Seamless integration with the EMPIR reconstruction workflow
- Configurable neutron source properties (energy, spatial distribution, etc.)
- Multi-process support for efficient data processing

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

2. Navigate to the cloned directory and install:
   ```bash
   cd G4LumaCam
   pip install .
   ```

3. Specify the folder containing the EMPIR executables in your environment:
   ```bash
   export EMPIR_PATH=/path/to/empir/executables
   ```

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
opm = lens.refocus(zfocus=25/1.58, zfine=13.3)
openbeam_data = lens.trace_rays(opm=opm, chunk_size=500, n_processes=10)

# Analyse and process data using EMPIR
o = lumacam.Analysis(archive="archive/test/openbeam").process_data()
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
