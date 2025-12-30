#!/usr/bin/env python3
"""Test detector_model groupby and TPX3 generation"""

import sys
from pathlib import Path
import shutil
from lumacam import Config, Simulate, Lens, VerbosityLevel

# Clean up old test
archive = "test_groupby"
if Path(archive).exists():
    shutil.rmtree(archive)

print("=" * 80)
print("TEST 1: Run simulation")
print("=" * 80)
config = Config.neutrons_tof(energy_min=1.0, energy_max=10.0)
config.num_events = 100  # Small number for testing

sim = Simulate(archive=archive)
sim.run(config)

# Check SimPhotons was created
simphotons = Path(archive) / "SimPhotons"
assert simphotons.exists(), "SimPhotons directory not created"
files = list(simphotons.glob("*.csv"))
print(f"✓ Created {len(files)} SimPhotons files")

print("\n" + "=" * 80)
print("TEST 2: Group by detector models")
print("=" * 80)
lens = Lens(archive=archive)

lens.groupby("detector_model", bins=[
    {"name": "test1", "detector_model": "image_intensifier", "blob": 2.0, "deadtime": 600},
    {"name": "test2", "detector_model": "image_intensifier_gain", "gain": 5000, "blob": 0, "deadtime": 475}
])

print(f"✓ Groupby configured")
print(f"  Mode: {getattr(lens, '_groupby_mode', 'NOT SET')}")
print(f"  Labels: {getattr(lens, '_groupby_labels', 'NOT SET')}")
print(f"  Configs: {getattr(lens, '_detector_model_configs', 'NOT SET')}")

print("\n" + "=" * 80)
print("TEST 3: Trace rays with verbose output")
print("=" * 80)

# Enable verbose output to see what's happening
try:
    result = lens.trace_rays(seed=42, verbosity=VerbosityLevel.DETAILED)
    print(f"✓ Trace rays completed")
    print(f"  Result type: {type(result)}")
    if result is not None:
        print(f"  Result length: {len(result)}")
except Exception as e:
    print(f"✗ Error during trace_rays: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 4: Check folder structure")
print("=" * 80)

detector_model_dir = Path(archive) / "detector_model"
if not detector_model_dir.exists():
    print(f"✗ detector_model directory not created at {detector_model_dir}")
    sys.exit(1)

print(f"✓ detector_model directory exists")

# Check each model folder
for model_name in ["test1", "test2"]:
    model_dir = detector_model_dir / model_name
    print(f"\n  Checking {model_name}:")

    if not model_dir.exists():
        print(f"    ✗ Model directory not created")
        continue

    # Check for SimPhotons
    simphotons = model_dir / "SimPhotons"
    if simphotons.exists():
        files = list(simphotons.glob("*.csv"))
        print(f"    ✓ SimPhotons: {len(files)} files")
    else:
        print(f"    ✗ SimPhotons not copied")

    # Check for tpx3Files
    tpx3_dir = model_dir / "tpx3Files"
    if tpx3_dir.exists():
        files = list(tpx3_dir.glob("*.tpx3"))
        print(f"    ✓ tpx3Files: {len(files)} files")
        if len(files) == 0:
            print(f"    ⚠ WARNING: tpx3Files directory exists but is empty!")
    else:
        print(f"    ✗ tpx3Files directory not created")

    # Check for TracedPhotons
    traced_dir = model_dir / "TracedPhotons"
    if traced_dir.exists():
        files = list(traced_dir.glob("*.csv"))
        print(f"    ✓ TracedPhotons: {len(files)} files")
    else:
        print(f"    ✗ TracedPhotons not created")

print("\n" + "=" * 80)
print("TEST 5: Check source detection")
print("=" * 80)

# Test source auto-detection directly
test_configs = [
    {"deadtime": 600, "blob": 2.0, "expected": "hits"},
    {"deadtime": 475, "blob": 0, "expected": "hits"},
    {"deadtime": None, "blob": 0, "expected": "photons"},
]

for config in test_configs:
    deadtime = config["deadtime"]
    blob = config["blob"]

    # Simulate auto-detection logic
    if (deadtime is not None and deadtime > 0) or blob > 0:
        detected_source = "hits"
    else:
        detected_source = "photons"

    expected = config["expected"]
    status = "✓" if detected_source == expected else "✗"
    print(f"  {status} deadtime={deadtime}, blob={blob} → {detected_source} (expected: {expected})")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Final check
all_passed = True
for model_name in ["test1", "test2"]:
    tpx3_dir = detector_model_dir / model_name / "tpx3Files"
    if tpx3_dir.exists():
        files = list(tpx3_dir.glob("*.tpx3"))
        if len(files) > 0:
            print(f"✓ {model_name}: {len(files)} TPX3 files generated")
        else:
            print(f"✗ {model_name}: tpx3Files directory exists but no files generated")
            all_passed = False
    else:
        print(f"✗ {model_name}: No tpx3Files directory")
        all_passed = False

if all_passed:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("\n❌ SOME TESTS FAILED - TPX3 files not generated")
    sys.exit(1)
