# Release Steps for v0.5.0

## 1. Push Branch

```bash
git push origin claude/enhance-photon-sensor-models-0S0lb
```

## 2. Create Pull Request

```bash
gh pr create \
  --title "Release v0.5.0: Advanced Detector Models & Elegant Model Comparison" \
  --body-file PR_DESCRIPTION.md \
  --base master \
  --head claude/enhance-photon-sensor-models-0S0lb
```

Or manually create PR at: https://github.com/TsvikiHirsh/G4LumaCam/pull/new/claude/enhance-photon-sensor-models-0S0lb

## 3. Review and Merge PR

Once approved, merge to master.

## 4. Create Release Tag

```bash
# Switch to master and pull
git checkout master
git pull origin master

# Create annotated tag
git tag -a v0.5.0 -m "Release v0.5.0: Advanced Detector Models & Elegant Model Comparison

Major features:
- 3 new physics-based detector models (image_intensifier_gain, timepix3_calibrated, physical_mcp)
- Elegant detector_model groupby API for model comparison
- Phosphor screen database (P20/P43/P46/P47)
- Export pixels functionality
- Comprehensive documentation

See CHANGELOG.md for full details."

# Push tag
git push origin v0.5.0
```

## 5. Create GitHub Release

```bash
gh release create v0.5.0 \
  --title "v0.5.0: Advanced Detector Models & Elegant Model Comparison" \
  --notes "$(cat <<'EOF'
## 🎯 Highlights

### Advanced Physics-Based Detector Models

Three new models for high-fidelity MCP+Timepix3 simulation:
- **image_intensifier_gain** (⭐ RECOMMENDED) - Gain-dependent blob sizing
- **timepix3_calibrated** - Logarithmic TOT response
- **physical_mcp** - Full MCP physics with phosphor types

### Elegant Model Comparison

\`\`\`python
lens.groupby(\"detector_model\", bins=[
    {\"name\": \"model1\", \"detector_model\": \"image_intensifier_gain\", \"gain\": 5000},
    {\"name\": \"model2\", \"detector_model\": \"physical_mcp\", \"phosphor_type\": \"p47\"}
])
lens.trace_rays(seed=42)  # Automatically processes both models
\`\`\`

### Phosphor Database

Auto-configuration for P20/P43/P46/P47 phosphor screens.

## 📚 Documentation

- [Detector Models Guide](.documents/DETECTOR_MODELS_SUMMARY.md)
- [Full Documentation](.documents/DETECTOR_MODELS.md)
- [Blob vs Gain Explained](.documents/BLOB_VS_GAIN.md)
- [Demo Notebook](notebooks/detector_models_comparison.ipynb)

## 📦 Installation

\`\`\`bash
git clone https://github.com/TsvikiHirsh/G4LumaCam.git
cd G4LumaCam
git checkout v0.5.0
pip install .
\`\`\`

## ⚠️ Breaking Changes

1. Default detector model: \`image_intensifier\` → \`image_intensifier_gain\`
2. Default decay_time: 10ns → 100ns

Migration: Explicitly specify old values if needed.

See [CHANGELOG.md](CHANGELOG.md) for complete details.
EOF
)"
```

Or manually at: https://github.com/TsvikiHirsh/G4LumaCam/releases/new?tag=v0.5.0

## 6. Verify Release

- [ ] PR merged to master
- [ ] Tag v0.5.0 created and pushed
- [ ] GitHub release published
- [ ] CHANGELOG.md updated
- [ ] README.md updated
- [ ] Documentation files in place

## Summary of Changes

**16 commits** merged:
- 3 new detector models
- Elegant groupby API for model comparison
- Phosphor database (P20/P43/P46/P47)
- Export pixels functionality
- Comprehensive documentation
- Multiple bug fixes

**Files changed:**
- Added: 4 documentation files, 1 demo notebook
- Modified: optics.py, analysis.py, setup.py, README.md, CHANGELOG.md
- Version: 0.4.0 → 0.5.0
