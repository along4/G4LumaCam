import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

class BuildGeant4Simulation(build_py):
    """Custom build step to compile the Geant4 simulation."""
    def run(self):
        # Create build directory
        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake and build
        subprocess.check_call(["cmake", "../src/G4LumaCam"], cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        # Path to the lumacam executable
        lumacam_executable = os.path.join(build_dir, "lumacam")

        # Debug output
        print("Build directory:", build_dir)
        print("Looking for lumacam executable at:", lumacam_executable)

        # Check if the executable exists
        if not os.path.exists(lumacam_executable):
            raise FileNotFoundError("lumacam executable not found in build directory.")

        # Create the bin directory in the package
        bin_dir = os.path.join(self.build_lib, "G4LumaCam", "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Copy the executable to the package directory
        subprocess.check_call(["cp", lumacam_executable, bin_dir])

        # Make the executable executable
        executable_path = os.path.join(bin_dir, "lumacam")
        os.chmod(executable_path, 0o755)

        # Continue with normal build process
        super().run()

class CustomInstall(install):
    """Custom install command to ensure the executable is properly installed."""
    def run(self):
        install.run(self)

setup(
    name="G4LumaCam",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        'G4LumaCam': ['bin/*'],  # Include binary files in the package
    },
    install_requires=[
        "rayoptics",
        "tqdm",
        "pandas",
        "scikit-learn"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lumacam=G4LumaCam.run_lumacam:main",
        ]
    },
    cmdclass={
        "build_py": BuildGeant4Simulation,
        "install": CustomInstall,
    },
)