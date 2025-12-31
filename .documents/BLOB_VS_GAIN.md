# Understanding Blob vs. Gain in Detector Models

## The Confusion

Both `blob` and `gain` affect blob size, but they work in **fundamentally different ways**:

## Simple Analogy

Think of it like a camera:
- **`gain`** = ISO setting (amplifies the signal, affects everything including blur)
- **`blob`** = Manual blur filter (just adds blur, doesn't affect signal strength)

---

## `gain` (Physics-Based, Dynamic)

### What it represents:
- **MCP electron multiplication factor**
- How many electrons are produced per input photon
- Physical parameter that depends on MCP voltage

### What it controls:
1. **Blob size** (automatic, physics-based): σ ∝ (gain)^0.4
2. **Signal strength** (more gain = more secondary photons)
3. **Noise characteristics** (gain affects SNR)

### How it works:
```
Input photon → MCP pore → Gain × electrons → Phosphor screen → Blob of photons
                           ↑
                    Higher gain = more electrons
                              = larger spatial spread
                              = bigger blob
```

### Example (image_intensifier_gain model):
```python
# Gain controls EVERYTHING automatically
lens.trace_rays(
    detector_model="image_intensifier_gain",
    gain=5000,       # ← Controls blob size via physics
    sigma_0=1.0,     # Base size at gain=1000
    blob=0           # Let gain control the blob!
)

# Result: σ = 1.0 × (5000/1000)^0.4 ≈ 2.09 pixels
```

Increasing gain from 1000 → 10000:
- ✅ Blob size increases (σ: 1.0 → 2.5 pixels)
- ✅ More photons per event
- ✅ Physically realistic
- ✅ Matches experimental behavior

---

## `blob` (Manual Override, Static)

### What it represents:
- **Manual blob size setting**
- User-specified spatial spread
- **Overrides any automatic calculation**

### What it controls:
1. **Blob size only** (fixed value)
2. Does NOT affect gain or signal strength
3. Just distributes photons spatially

### How it works:
```
Input photon → blob parameter → Distribute photons in circle/Gaussian of size=blob
                    ↑
             User says "make blob this big"
             (ignores physics)
```

### Example (simple image_intensifier model):
```python
# Blob is a fixed manual setting
lens.trace_rays(
    detector_model="image_intensifier",  # Simple model
    blob=2.0,        # ← Fixed blob radius
    deadtime=600
)

# Result: Every photon creates a blob of exactly 2.0 pixels radius
```

Increasing blob from 1.0 → 5.0:
- ✅ Blob size increases
- ❌ No physical meaning (arbitrary)
- ❌ Same number of photons (just spread differently)
- ❌ Doesn't match gain changes

---

## When to Use Each

### Use `gain` (Recommended):

**For realistic MCP simulations:**
```python
lens.trace_rays(
    detector_model="image_intensifier_gain",  # ⭐ Physics-based
    gain=5000,              # Your MCP voltage setting
    sigma_0=1.0,            # Base blob at reference gain
    blob=0                  # Let gain control blob size
)
```

**Why:**
- ✅ Physically accurate
- ✅ Matches experimental data
- ✅ Blob size scales with gain (σ ∝ gain^0.4)
- ✅ Can compare different MCP voltages
- ✅ Recommended for Timepix3

### Use `blob` (Manual Control):

**For simple simulations or testing:**
```python
lens.trace_rays(
    detector_model="image_intensifier",  # Simple model
    blob=2.0,               # Fixed blob size
    deadtime=600
)
```

**Why:**
- ✅ Simple and fast
- ✅ Good for quick tests
- ✅ When you don't care about MCP physics
- ❌ Not physically realistic
- ❌ Can't model gain variations

---

## What Happens When You Use Both?

### In `image_intensifier_gain` model:

```python
lens.trace_rays(
    detector_model="image_intensifier_gain",
    gain=5000,       # Wants to make σ = 2.09 pixels
    sigma_0=1.0,
    blob=3.0         # ← OVERRIDES gain calculation!
)

# Result: Blob = 3.0 pixels (gain is ignored!)
```

**The override logic:**
```python
# Internal code
sigma_pixels = sigma_0 * (gain / gain_ref) ** gain_exponent  # Calculate from gain

if blob > 0:
    sigma_pixels = blob  # ← blob overrides everything!
```

---

## Model Comparison

| Model | Blob Control | Gain Control | Best For |
|-------|--------------|--------------|----------|
| `image_intensifier` | `blob` (manual) | ❌ No gain | Simple tests |
| `image_intensifier_gain` | `gain` (physics) | ✅ Yes | **Timepix3 + MCP** ⭐ |
| `physical_mcp` | `gain` (physics) | ✅ Yes | High-fidelity sims |
| `gaussian_diffusion` | `blob` (sigma) | ❌ No gain | CCD/CMOS sensors |

---

## Physics Behind Gain-Dependent Blob

### Why does blob size scale with gain?

1. **Electron multiplication** in MCP pores
   - Higher voltage → more electrons
   - Electrons spread spatially as they multiply

2. **Phosphor screen emission**
   - More electrons → larger impact area
   - Photon distribution is Gaussian

3. **Empirical scaling law** (from literature):
   ```
   σ = σ₀ × (gain / gain_ref)^α
   ```
   Where α ≈ 0.4 (experimentally determined)

4. **Reference:**
   - Siegmund et al.: MCP spatial resolution studies
   - Photonis MCP specifications
   - Typical: α = 0.3-0.5

### Real-world example:

| MCP Voltage | Gain | Expected σ (pixels) |
|-------------|------|---------------------|
| 800V | 1000 | 1.0 |
| 900V | 2000 | 1.32 |
| 1000V | 5000 | 2.09 |
| 1100V | 10000 | 2.51 |
| 1200V | 20000 | 3.31 |

---

## Recommendations

### For Most Users (Timepix3 + MCP):

```python
# USE THIS:
lens.trace_rays(
    detector_model="image_intensifier_gain",  # ⭐ Recommended
    gain=5000,              # Adjust to your MCP voltage
    sigma_0=1.0,            # Typical base blob
    blob=0,                 # CRITICAL: Let gain control blob!
    deadtime=475            # TPX3 spec
)
```

### For Quick Tests:

```python
# OR THIS (simpler):
lens.trace_rays(
    detector_model="image_intensifier",
    blob=2.0,               # Fixed blob
    deadtime=600
)
```

---

## Summary

| Parameter | Purpose | When to Use | Physics-Based? |
|-----------|---------|-------------|----------------|
| **`gain`** | MCP electron multiplication | MCP simulations | ✅ Yes |
| **`blob`** | Manual blob size override | Simple tests | ❌ No |

**Golden Rule:** For realistic MCP simulations, use `detector_model="image_intensifier_gain"` with `gain` parameter and `blob=0`!
