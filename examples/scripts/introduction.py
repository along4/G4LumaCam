# This script serves as an introduction to using the G4LumaCam package. 
# It demonstrates how to set up a simulation, configure it, and run it 
# to obtain results. The script is structured in a way that allows users 
# to easily follow along and understand the key components of the simulation 
# process.

from lumacam.simulations.g4config import Config
from lumacam.simulations.g4runner import Simulate

# Create a configuration for the simulation. This includes setting parameters
# such as the type of particles, the materials involved, and the geometry of the setup.
config = Config.dt_neutrons_white()
print(config)