# G4LumaCam v0.4.0 Release Notes

We're excited to announce the release of **G4LumaCam v0.4.0**! This release brings major improvements to optical modeling, analysis capabilities, and TPX3 file generation, along with a comprehensive tutorial.

## Highlights

### New Tutorial
- **[Comprehensive Tutorial Notebook](notebooks/G4LumaCam_Tutorial.ipynb)**: Step-by-step guide covering simulation setup, optical ray tracing, and data analysis workflows

### Enhanced Optical System
- **Improved Lens Modeling**: Support for ZMX lens files and multiple lens configurations
- **Better Focus Control**: New `zfine` parameter for precise focus adjustments
- **Lens Examples**: Added data files for JP1987-249119 and JP2000-019398 lens models

### Advanced Analysis Tools
- **MTF Analysis**: Calculate Modulation Transfer Function for optical system characterization
- **ROI Analysis**: Region of Interest analysis with comprehensive statistics
- **Event Tracking**: Full neutron-to-pixel traceability with `neutron_id` column

### TPX3 File Generation
- **Standard Output Format**: Generate TPX3 files compatible with EMPIR and other Timepix-3 reconstruction tools
- **Realistic Detector Effects**: Configurable saturation, deadtime (default 600ns), and blob modeling
- **Event Organization**: Support for event-based file organization

### Pulsed Sources
- **Pulsed Neutron Sources**: Configure frequency and flux parameters
- **Time Spread Control**: Uniform or custom time distributions
- **Multiple Particle Types**: Support for neutrons, ions, and gamma sources

### Scintillator Options
- **LYSO Support**: Added LYSO scintillator material
- **Configurable Thickness**: Adjust scintillator and sample thickness
- **EJ-200 Defaults**: Optimized default properties for EJ-200 scintillator

### Flexible EMPIR Integration
Three convenient ways to configure EMPIR paths:
1. Environment variable during installation
2. Runtime parameter in Analysis class
3. Default `./empir` fallback path

## Breaking Changes

- **Lens Focus Parameter**: The `focus` parameter has been renamed to `zfine` for improved clarity

## Key Bug Fixes

- Fixed TPX3 coordinate system (resolves y-axis pixel mapping)
- Corrected time-of-flight calculations with proper trigger handling
- Fixed ZMX lens file loading issues
- Resolved scintillator geometry update problems

## Installation

```bash
git clone https://github.com/TsvikiHirsh/G4LumaCam.git
cd G4LumaCam
pip install .
```

Optional EMPIR configuration:
```bash
export EMPIR_PATH=/path/to/empir/executables
pip install .
```

## Quick Start

```python
import lumacam

# 1. Run simulation
sim = lumacam.Simulate("openbeam")
config = lumacam.Config.neutrons_uniform_energy()
df = sim.run(config)

# 2. Trace rays and generate TPX3 files
lens = lumacam.Lens(archive="openbeam")
lens.trace_rays(blob=1.0, deadtime=600)

# 3. Reconstruct with EMPIR (optional)
analysis = lumacam.Analysis(archive="archive/test/openbeam")
analysis.process(params="hitmap", event2image=True)
```

## Documentation

- **[Tutorial Notebook](notebooks/G4LumaCam_Tutorial.ipynb)**: Complete walkthrough with examples
- **[README](README.md)**: Installation and usage guide
- **[CHANGELOG](CHANGELOG.md)**: Detailed list of all changes

## What's Next

Future releases will focus on:
- Additional scintillator materials
- Performance optimizations for large-scale simulations
- Enhanced visualization tools
- More example configurations

## Acknowledgments

Thank you to all users who provided feedback and helped improve G4LumaCam!

## Support

- **Issues**: [GitHub Issues](https://github.com/TsvikiHirsh/G4LumaCam/issues)
- **Repository**: [https://github.com/TsvikiHirsh/G4LumaCam](https://github.com/TsvikiHirsh/G4LumaCam)

---

**Full Changelog**: See [CHANGELOG.md](CHANGELOG.md) for complete details.
