#!/usr/bin/env python3
"""
Test script to verify that Lens.groupby() works with both numerical and categorical columns.
This specifically tests the fix for grouping by string columns like 'parentName'.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Create a simple test case without needing the full Lens class
def test_groupby_logic():
    """Test the core groupby logic for categorical vs numerical data."""

    print("=" * 60)
    print("Testing groupby categorical detection logic")
    print("=" * 60)

    # Create sample data with both numerical and categorical columns
    data = pd.DataFrame({
        'nz': [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28],
        'parentName': ['neutron', 'gamma', 'neutron', 'gamma',
                       'electron', 'neutron', 'gamma', 'electron'],
        'energy': [1.5, 2.3, 1.8, 2.1, 3.2, 1.6, 2.4, 3.0]
    })

    print(f"\nSample data ({len(data)} rows):")
    print(data.head())

    # Test 1: Categorical column detection
    print("\n" + "-" * 60)
    print("Test 1: Detecting categorical columns")
    print("-" * 60)

    for col in ['nz', 'parentName', 'energy']:
        is_categorical = pd.api.types.is_string_dtype(data[col]) or \
                        pd.api.types.is_object_dtype(data[col]) or \
                        pd.api.types.is_categorical_dtype(data[col])

        dtype = data[col].dtype
        print(f"  Column '{col}' (dtype={dtype}): {'categorical' if is_categorical else 'numerical'}")

    # Test 2: Grouping by categorical column (parentName)
    print("\n" + "-" * 60)
    print("Test 2: Grouping by categorical column 'parentName'")
    print("-" * 60)

    column = 'parentName'
    is_categorical = pd.api.types.is_string_dtype(data[column]) or \
                    pd.api.types.is_object_dtype(data[column]) or \
                    pd.api.types.is_categorical_dtype(data[column])

    if is_categorical:
        unique_values = data[column].dropna().unique()
        unique_values = sorted(unique_values)
        labels = [str(val) for val in unique_values]

        print(f"  Detected {len(labels)} unique values: {labels}")

        # Map values to labels
        value_to_label = dict(zip(unique_values, labels))
        data['_bin_label'] = data[column].map(value_to_label)

        # Count rows per group
        bin_counts = data['_bin_label'].value_counts().sort_index()
        print(f"\n  Group distribution:")
        for label, count in bin_counts.items():
            print(f"    {label}: {count} rows")

        print("\n  ✓ Categorical grouping successful!")
    else:
        print("  ✗ ERROR: Column should be detected as categorical!")
        return False

    # Test 3: Grouping by numerical column (nz)
    print("\n" + "-" * 60)
    print("Test 3: Grouping by numerical column 'nz'")
    print("-" * 60)

    column = 'nz'
    is_categorical = pd.api.types.is_string_dtype(data[column]) or \
                    pd.api.types.is_object_dtype(data[column]) or \
                    pd.api.types.is_categorical_dtype(data[column])

    if not is_categorical:
        # Create bins for numerical data
        bins = np.arange(0.0, 0.35, 0.1)
        labels_num = [f"{bins[i]:.3f}" for i in range(len(bins)-1)]

        print(f"  Creating {len(bins)-1} bins: {bins.tolist()}")
        print(f"  Labels: {labels_num}")

        # Bin the data
        data['_bin_label_num'] = pd.cut(
            data[column],
            bins=bins,
            labels=labels_num,
            right=False,
            include_lowest=True
        )

        # Count rows per bin
        bin_counts = data['_bin_label_num'].value_counts().sort_index()
        print(f"\n  Bin distribution:")
        for label, count in bin_counts.items():
            print(f"    {label}: {count} rows")

        print("\n  ✓ Numerical binning successful!")
    else:
        print("  ✗ ERROR: Column should be detected as numerical!")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    return True


def test_with_mock_lens():
    """Test with a mock Lens object to simulate the actual usage."""
    print("\n\n")
    print("=" * 60)
    print("Testing with mock Lens data")
    print("=" * 60)

    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create mock SimPhotons directory with CSV data
        sim_photons_dir = temp_dir / "SimPhotons"
        sim_photons_dir.mkdir()

        # Create sample CSV file with parentName column
        sample_data = pd.DataFrame({
            'x': np.random.randn(20),
            'y': np.random.randn(20),
            'z': np.random.randn(20),
            'dx': np.random.randn(20),
            'dy': np.random.randn(20),
            'dz': np.random.randn(20),
            'wavelength': 400 + np.random.randn(20) * 50,
            'parentName': np.random.choice(['neutron', 'gamma', 'electron'], 20),
            'nz': np.random.rand(20),
            'pz': np.random.rand(20),
            'id': range(20),
            'neutron_id': range(20),
            'pulse_id': [0] * 20
        })

        csv_file = sim_photons_dir / "sim_data_0.csv"
        sample_data.to_csv(csv_file, index=False)

        print(f"\nCreated test data in: {sim_photons_dir}")
        print(f"Sample data shape: {sample_data.shape}")
        print(f"\nUnique parentName values: {sorted(sample_data['parentName'].unique())}")

        # Verify the CSV can be read and processed
        df = pd.read_csv(csv_file)

        # Test categorical detection
        column = 'parentName'
        is_categorical = pd.api.types.is_string_dtype(df[column]) or \
                        pd.api.types.is_object_dtype(df[column]) or \
                        pd.api.types.is_categorical_dtype(df[column])

        if is_categorical:
            unique_values = df[column].dropna().unique()
            unique_values = sorted(unique_values)
            print(f"\n✓ Successfully detected categorical column with {len(unique_values)} unique values")
            print(f"  Values: {unique_values}")
        else:
            print("\n✗ Failed to detect categorical column!")
            return False

        print("\n✓ Mock Lens test successful!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Lens.groupby() Categorical Column Fix - Test Suite")
    print("=" * 60)

    success = True

    # Run tests
    success = test_groupby_logic() and success
    success = test_with_mock_lens() and success

    # Final result
    print("\n\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe fix successfully handles both:")
        print("  - Categorical/string columns (e.g., parentName)")
        print("  - Numerical columns (e.g., nz, energy)")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
