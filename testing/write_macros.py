import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.append(str(src_path))

from lumacam.simulate import Config

# Create a configuration (e.g., uniform neutron source)
config = Config.neutrons_uniform()

# Specify the output macro file path
output_macro_path = "/home/l280162/Simulations/G4LumaCam/output/macro.mac"

# Write the macro file
config.write(output_macro_path)

print(f"Macro file written to: {output_macro_path}")