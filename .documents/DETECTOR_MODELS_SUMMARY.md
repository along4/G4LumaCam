# Detector Models Summary - G4LumaCam

## Overview

G4LumaCam provides **8 physics-based detector models** for simulating image intensifiers, MCPs, and event cameras. This document summarizes the models and provides quick-start guidance.

## Quick Reference

### Model Hierarchy (by sophistication)

1. **`direct_detection`** - Simplest: No blob, fast computation
2. **`image_intensifier`** - Simple: Uniform circular blob
3. **`gaussian_diffusion`** - CCD/CMOS charge diffusion
4. **`wavelength_dependent`** - Spectral response
5. **`avalanche_gain`** - Poisson statistics (APD/PMT)
6. **`image_intensifier_gain`** ⭐ - **RECOMMENDED** for Timepix3
7. **`timepix3_calibrated`** - Calibrated TOT curve
8. **`physical_mcp`** - Most sophisticated: Full MCP physics

---

## For Timepix3 Users (RECOMMENDED)

### Standard Setup: `image_intensifier_gain`

```python
from lumacam import Lens

lens = Lens(archive="my_simulation")

# Typical Timepix3 + MCP intensifier
lens.trace_rays(
    detector_model="image_intensifier_gain",
    gain=5000,              # MCP gain @ 1000V
    sigma_0=1.0,            # Base blob size (pixels)
    gain_exponent=0.4,      # From MCP literature
    decay_time=100.0,       # P47 phosphor (~100ns)
    deadtime=475.0,         # Timepix3 spec
    blob=0                  # IMPORTANT: blob=0 for gain-dependent calculation!
)
```

⚠️ **CRITICAL:** Set `blob=0` to enable automatic gain-dependent blob calculation. If `blob > 0`, it overrides the gain!

**Why this model?**
- ✅ Gain-dependent blob scaling: σ ∝ (gain)^0.4
- ✅ Gaussian photon distribution (realistic)
- ✅ Based on Photonis specs and Siegmund et al.
- ✅ Matches Timepix3 deadtime (475ns)

### Chevron MCP (Higher gain)

```python
lens.trace_rays(
    detector_model="image_intensifier_gain",
    gain=10000,             # Higher gain @ 1200V
    sigma_0=0.8,            # Smaller base (better resolution)
    decay_time=70.0,        # Fast P47 component
    deadtime=475.0,
    blob=0                  # Enable gain-dependent blob!
)
```

---

## For Experimental Matching: `physical_mcp`

### P47 Phosphor (Modern, Fast)

```python
lens.trace_rays(
    detector_model="physical_mcp",
    gain=8000,                    # Chevron MCP
    gain_noise_factor=1.3,        # Realistic noise
    phosphor_type='p47',          # Fast YAG:Ce
    # Decay parameters auto-set for P47:
    # decay_fast=70ns, decay_slow=200ns, fast_fraction=0.9
    deadtime=475.0,
    blob=1
)
```

### P43 Phosphor (Traditional Gen 2/3)

```python
lens.trace_rays(
    detector_model="physical_mcp",
    gain=5000,
    phosphor_type='p43',
    # Auto-set: decay_fast=100ns, decay_slow=1000ns, fast_fraction=0.6
    deadtime=600.0,
    blob=1
)
```

### Custom Phosphor Parameters

```python
lens.trace_rays(
    detector_model="physical_mcp",
    gain=5000,
    phosphor_type='custom',  # Or use p20/p43/p46/p47
    decay_fast=80.0,         # Your measured fast decay
    decay_slow=250.0,        # Your measured slow decay
    fast_fraction=0.85,      # Fraction in fast component
    deadtime=475.0,
    blob=1
)
```

---

## Phosphor Types (for `physical_mcp` model)

| Phosphor | Material | Decay Time | Color | Use Case |
|----------|----------|------------|-------|----------|
| **P47** | Y₃Al₅O₁₂:Ce (YAG:Ce) | 70-100 ns | Yellow | Modern Chevron MCPs, fast imaging |
| **P46** | Y₂SiO₅:Ce | ~70 ns | Blue | High frame rate |
| **P43** | Gd₂O₂S:Tb | ~1 ms | Yellow-green | Traditional Gen 2/3 |
| **P20** | ZnCdS:Ag | 100 ns + 1ms tail | Green | Legacy systems |

**Default in `physical_mcp`:** P47 (modern fast phosphor)

---

## Calibrated Timepix3: `timepix3_calibrated`

Use this when you have **calibration data** from your actual Timepix3 detector.

```python
# With measured calibration
lens.trace_rays(
    detector_model="timepix3_calibrated",
    gain=5000,
    sigma_pixels=1.5,
    tot_a=28.5,              # Your calibrated a (from measurement)
    tot_b=52.3,              # Your calibrated b (from measurement)
    deadtime=475.0,
    pixel_variation=0.03,    # Per-pixel variation (3%)
    blob=1
)
```

**TOT Model:** TOT = a + b × ln(Q/Q_ref)
Based on Poikela et al. 2014 (Timepix3 paper)

---

## Key Parameters Explained

### gain
- **Physical meaning:** MCP electron multiplication factor
- **Typical values:** 1000-20000
- **Depends on:** MCP voltage (higher V → higher gain)
- **Effect:** Higher gain → larger blob size (σ ∝ gain^0.4)

### decay_time
- **Physical meaning:** Phosphor emission decay time constant
- **Typical values:**
  - P47: 70-100 ns (fast)
  - P43: ~1000 ns (slow)
  - P46: ~70 ns (very fast)
- **Effect:** Controls photon arrival time distribution

### deadtime
- **Physical meaning:** Pixel inactive time after activation
- **Typical values:**
  - Timepix3: 475 ns (spec)
  - Generic event cameras: 300-600 ns
- **Effect:** Photons within deadtime update TOT, don't create new events

### blob
- **For `image_intensifier`:** Maximum blob radius (pixels)
- **For `image_intensifier_gain`:** Overridden by gain-dependent calculation
- **For `gaussian_diffusion`:** Gaussian sigma (pixels)
- **Effect:** Spatial spreading of photon response

---

## Model Comparison Table

| Model | Blob Type | Gain Control | Phosphor Decay | Deadtime | Best For |
|-------|-----------|--------------|----------------|----------|----------|
| `image_intensifier` | Uniform circle | Fixed | Single exp | Custom | Simple MCP |
| `image_intensifier_gain` ⭐ | Gaussian, gain-dependent | ✅ Variable | Single exp | 475ns | **Timepix3 + MCP** |
| `timepix3_calibrated` | Gaussian | ✅ Variable | None | 475ns | **Calibrated TPX3** |
| `physical_mcp` | Gaussian, gain-dependent | ✅ Variable | Bi-exponential | Custom | **High-fidelity matching** |
| `gaussian_diffusion` | Gaussian | None | None | Custom | CCD/CMOS |
| `direct_detection` | Single pixel | None | None | Custom | Fast computation |

---

## Complete Example: Model Comparison

See `/notebooks/detector_models_demo.ipynb` for a complete working example that:

1. Runs a `Config.neutrons_tof()` simulation
2. Tests all 8 detector models
3. Compares TOT distributions
4. Visualizes gain scaling
5. Provides performance metrics

### Quick Start:

```python
from lumacam import Config, Lens

# 1. Run simulation
config = Config.neutrons_tof(energy_min=1.0, energy_max=10.0)
config.n_events = 100
config.archive = "demo_models"
config.run()

# 2. Test different models
lens = Lens(archive="demo_models")

# Model 1: Simple
lens.trace_rays(detector_model="image_intensifier", blob=2.0, deadtime=600, seed=42)

# Model 2: Recommended for Timepix3
lens.trace_rays(detector_model="image_intensifier_gain", gain=5000, deadtime=475, seed=42)

# Model 3: Full physics
lens.trace_rays(detector_model="physical_mcp", gain=5000, phosphor_type='p47', deadtime=475, seed=42)
```

---

## References

1. **Poikela et al. 2014** - "Timepix3: a 65K channel hybrid pixel readout chip with simultaneous ToA/ToT and sparse readout"
   DOI: 10.1088/1748-0221/9/05/C05013

2. **Siegmund et al.** - "Microchannel plate imaging detectors for UV and charged particle detection"
   Multiple papers on MCP physics and gain statistics

3. **Photonis Technical Documentation** - MCP specifications, gain curves, phosphor characteristics

4. **Chevron MCP Design** - Higher gain, better signal-to-noise, used in modern intensifiers

---

## Support

- **Full documentation:** [DETECTOR_MODELS.md](DETECTOR_MODELS.md)
- **Tutorial notebook:** `/notebooks/detector_models_demo.ipynb`
- **Example configs:** `/src/lumacam/config/empir_params.py`
- **Issues:** https://github.com/anthropics/G4LumaCam/issues

---

## Changelog

### v0.4.0 (2025-12-30)
- ✅ Added `image_intensifier_gain` model (gain-dependent blob)
- ✅ Added `timepix3_calibrated` model (logarithmic TOT)
- ✅ Added `physical_mcp` model (full physics, multi-exp decay)
- ✅ Added P47 phosphor support (default in `physical_mcp`)
- ✅ Added phosphor database (P20/P43/P46/P47)
- ✅ Updated documentation and examples

### v0.3.0 (Previous)
- Basic detector models (5 models)
- Simple image intensifier
- Gaussian diffusion, direct detection
- Wavelength dependent, avalanche gain
