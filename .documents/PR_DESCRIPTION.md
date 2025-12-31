# Release v0.5.0: Advanced Detector Models & Elegant Model Comparison

## 🎯 Summary

This PR introduces advanced physics-based detector models for high-fidelity MCP+Timepix3 simulations, an elegant `groupby` API for comparing detector models, and comprehensive documentation.

## ⭐ Major Features

### 1. Advanced Detector Models (3 new models)

- **`image_intensifier_gain`** (⭐ **RECOMMENDED** for Timepix3+MCP)
  - Gain-dependent blob sizing: σ ∝ gain^0.4
  - Based on Photonis specs and Siegmund et al.
  - Realistic MCP physics

- **`timepix3_calibrated`**
  - Logarithmic TOT response: TOT = a + b × ln(Q)
  - Per-pixel variation support
  - Calibrated to real Timepix3 data

- **`physical_mcp`**
  - Full MCP physics with Poisson gain statistics
  - Bi-exponential phosphor decay
  - Support for P20/P43/P46/P47 phosphors

### 2. Elegant Model Comparison API

```python
# Define multiple detector models
lens.groupby("detector_model", bins=[
    {"name": "intensifier", "detector_model": "image_intensifier", "blob": 2.0, "deadtime": 600},
    {"name": "full_physics", "detector_model": "physical_mcp", "gain": 8000, "phosphor_type": "p47", "blob": 0}
])

# Trace all models automatically (separate folders)
lens.trace_rays(seed=42)

# Process all models automatically
analysis = Analysis(archive=f"{archive}/detector_model")
analysis.process(params="hitmap", export_photons=True, export_events=True)
```

**Clean folder structure:**
```
archive/
├── SimPhotons/
└── detector_model/
    ├── intensifier/
    │   ├── tpx3Files/
    │   ├── ExportedPhotons/
    │   └── ExportedEvents/
    └── full_physics/
        └── ...
```

### 3. Phosphor Screen Database

Auto-configuration for 4 phosphor types:
- **P47** (YAG:Ce): 70-100ns, modern Chevron MCPs (default)
- **P46** (Y₂SiO₅:Ce): ~70ns, high frame rate
- **P43** (Gd₂O₂S:Tb): ~1ms, traditional Gen 2/3
- **P20** (ZnCdS:Ag): 100ns + 1ms tail, legacy systems

### 4. Export Pixels Functionality

```python
analysis.process(
    export_photons=True,
    export_events=True,
    export_pixels=True  # ← NEW
)
```

Generates CSV files in `ExportedPixels/` for detailed pixel activation analysis.

## 📚 Documentation

### New Documentation Files

- **[DETECTOR_MODELS.md](.documents/DETECTOR_MODELS.md)** - Complete guide to all 8 detector models
- **[DETECTOR_MODELS_SUMMARY.md](.documents/DETECTOR_MODELS_SUMMARY.md)** - Quick reference
- **[BLOB_VS_GAIN.md](.documents/BLOB_VS_GAIN.md)** - Physics explanation of blob vs gain parameters
- **[detector_models_comparison.ipynb](notebooks/detector_models_comparison.ipynb)** - Interactive demo

### Updated Documentation

- **README.md** - Updated with detector models info and examples
- **CHANGELOG.md** - Complete v0.5.0 release notes

## 🔧 Implementation Details

### API Changes

1. **Default detector model**: `image_intensifier` → `image_intensifier_gain`
2. **Default decay_time**: 10ns → 100ns (P47 phosphor standard)
3. **Auto-detect TPX3 generation**: Source auto-detects as 'hits' when deadtime > 0 or blob > 0
4. **Suffix parameter**: Added to trace_rays() for organized outputs
5. **Detector model groupby**: Extended groupby() to accept detector configurations

### Key Files Modified

- `src/lumacam/optics.py` - Detector models, groupby, auto-detection
- `src/lumacam/analysis.py` - Export pixels, auto-process grouped structures
- `src/lumacam/config/empir_params.py` - Updated parameters
- `setup.py` - Version bump to 0.5.0

## 🧪 Testing

Comprehensive test suite validates:
- ✅ Detector model groupby creates proper folder structure
- ✅ Source auto-detection works correctly
- ✅ TPX3 files generated for each model
- ✅ Analysis.process() auto-detects and processes all groups
- ✅ All 8 detector models work correctly
- ✅ Gain scaling verified (σ ∝ gain^0.4)

## 🐛 Bug Fixes

1. **NameError**: Fixed detector_model parameter passing
2. **TypeError**: Avoid duplicate keyword arguments in detector_model groupby
3. **TPX3 generation**: Fixed auto-detection to ensure TPX3 files are generated
4. **Gain scaling**: Fixed blob=0 requirement for gain-dependent models

## 📋 Breaking Changes

⚠️ **Minor breaking changes:**

1. Default detector model changed to `image_intensifier_gain`
   - Old behavior: `blob=1` (fixed size)
   - New behavior: `blob=0, gain=5000` (physics-based)
   - **Migration**: Explicitly specify `detector_model="image_intensifier"` for old behavior

2. Default `decay_time` changed from 10ns to 100ns
   - **Migration**: Explicitly specify `decay_time=10` if needed

**Backward compatibility maintained** for all existing detector models.

## 🚀 What's Next

After merge:
1. Tag release: `git tag v0.5.0`
2. Push tag: `git push origin v0.5.0`
3. Create GitHub release with CHANGELOG

## 📊 Metrics

- **16 commits** with comprehensive improvements
- **3 new detector models** (total: 8)
- **4 documentation files** created
- **1 interactive demo notebook**
- **Full test coverage** with validation suite

---

**Recommended for Timepix3 users!** 🎉

This release provides the most realistic MCP+Timepix3 simulation to date, with physics-based gain control and phosphor decay modeling.
