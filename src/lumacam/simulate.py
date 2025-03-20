import os
import subprocess
import re
from dataclasses import dataclass
from importlib import resources
import shutil
from typing import Optional, Tuple
from pathlib import Path
from enum import IntEnum
from tqdm.notebook import tqdm
import pandas as pd
import threading
import queue
import time

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

@dataclass
class Config:
    """Configuration for Geant4 simulation."""
    # GPS parameters
    particle: str = "neutron"
    energy: float = 10.0
    energy_unit: str = "MeV"
    energy_type: str = "Mono"  # Can be "Mono" or "Lin" for linear distribution
    energy_min: Optional[float] = None  # Used for Lin distribution
    energy_max: Optional[float] = None  # Used for Lin distribution
    energy_gradient: Optional[float] = None  # Used for Lin distribution
    energy_intercept: Optional[float] = None  # Used for Lin distribution
    position_x: float = 0
    position_y: float = 0
    position_z: float = -1059
    position_unit: str = "cm"
    direction_x: float = 0
    direction_y: float = 0
    direction_z: float = 1
    shape: str = "Rectangle"
    halfx: float = 60
    halfy: float = 60
    shape_unit: str = "mm"
    angle_type: str = "iso"
    max_theta: float = 0
    min_theta: float = 0
    angle_unit: str = "deg"
    sample_material: str = "G4_Galactic" # Material of the sample
    csv_batch_size: int = 0
    
    # Run parameters
    num_events: int = 100000
    progress_interval: int = 100
    csv_filename: str = "sim_data.csv"

    @classmethod
    def neutrons_uniform(cls) -> 'Config':
        """Default neutron configuration with uniform distribution."""
        return cls(
            particle="neutron",
            energy=10.0,
            energy_unit="MeV",
            position_z=-1059,
            position_unit="cm",
            halfx=60,
            halfy=60,
            shape_unit="mm",
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_Graphite",
            csv_batch_size=0,
        )
    
    @classmethod
    def neutrons_uniform_energy(cls) -> 'Config':
        """Neutron configuration with uniform spatial and energy distribution."""
        return cls(
            particle="neutron",
            energy_type="Lin",
            energy_min=2.0,
            energy_max=10.0,
            energy_gradient=0.0,
            energy_intercept=1.0,
            energy_unit="MeV",
            position_z=-1059,
            position_unit="cm",
            halfx=60,
            halfy=60,
            shape_unit="mm",
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_Graphite",
            csv_batch_size=0,
        )


    @classmethod
    def opticalphoton_point(cls) -> 'Config':
        """Point source optical photon configuration."""
        return cls(
            particle="opticalphoton",
            energy=3,
            energy_unit="eV",
            position_z=20.,
            position_unit="mm",
            halfx=0.0001,
            halfy=0.0001,
            shape_unit="um",
            num_events=10000,
            max_theta=180,
            min_theta=177,
            progress_interval=1000,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            csv_batch_size=1000,
        )

    @classmethod
    def opticalphoton_uniform(cls) -> 'Config':
        """Uniform source optical photon configuration."""
        return cls(
            particle="opticalphoton",
            energy=3,
            energy_unit="eV",
            position_z=20,
            position_unit="mm",
            halfx=60,
            halfy=60,
            max_theta=180,
            min_theta=177,
            shape_unit="mm",
            num_events=100000,
            progress_interval=1000,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            csv_batch_size=1000,
        )

    def write(self, output_file: str) -> str:
        """
        Write configuration to a Geant4 macro file.
        
        Args:
            output_file: Path where the macro file will be written
            
        Returns:
            Path to the created macro file
        """
        macro_content = f"""
/gps/particle {self.particle}
"""
        if self.energy_type == "Mono":
            macro_content += f"/gps/energy {self.energy} {self.energy_unit}\n"
        elif self.energy_type == "Lin":
            macro_content += f"""
/gps/ene/type Lin
/gps/ene/min {self.energy_min} {self.energy_unit}
/gps/ene/max {self.energy_max} {self.energy_unit}
/gps/ene/gradient {self.energy_gradient}
/gps/ene/intercept {self.energy_intercept}
"""
        
        macro_content += f"""
/gps/position {self.position_x} {self.position_y} {self.position_z} {self.position_unit}
/gps/direction {self.direction_x} {self.direction_y} {self.direction_z}
/gps/pos/shape {self.shape}
/gps/pos/halfx {self.halfx} {self.shape_unit}
/gps/pos/halfy {self.halfy} {self.shape_unit}
/gps/pos/type Plane
/gps/ang/type {self.angle_type}
/gps/ang/maxtheta {self.max_theta} {self.angle_unit}
/gps/ang/mintheta {self.min_theta} {self.angle_unit}
/run/printProgress {self.progress_interval}
/lumacam/sampleMaterial {self.sample_material}
/lumacam/batchSize {self.csv_batch_size}
/run/beamOn {self.num_events}
"""
        # Ensure the directory exists
        # Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Write the macro file
        with open(output_file, 'w') as f:
            f.write(macro_content.strip())
            
        return output_file

    def __str__(self) -> str:
        """Return a human-readable string representation of the configuration."""
        energy_info = ""
        if self.energy_type == "Mono":
            energy_info = f"Energy: {self.energy} {self.energy_unit}\n"
        elif self.energy_type == "Lin":
            energy_info = (f"Energy: uniform distribution from {self.energy_min} to {self.energy_max} {self.energy_unit}\n"
                          f"  (gradient: {self.energy_gradient}, intercept: {self.energy_intercept})\n")
            
        return (
            f"Configuration:\n"
            f"  Particle: {self.particle}\n"
            f"  {energy_info}"
            f"  Position: ({self.position_x}, {self.position_y}, {self.position_z}) {self.position_unit}\n"
            f"  Direction: ({self.direction_x}, {self.direction_y}, {self.direction_z})\n"
            f"  Shape: {self.shape} ({self.halfx}x{self.halfy} {self.shape_unit})\n"
            f"  Angle: {self.angle_type} (max theta: {self.max_theta} {self.angle_unit})\n"
            f"  Events: {self.num_events}\n"
            f"  Output: {self.csv_filename}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return str(self)



class Simulate:
    """Class to simulate the lumacam executable."""
    def __init__(self, archive: str = "archive/test"):
        """
        Initialize the Simulate object.

        Input:
            archive: Path to the archive directory
        """
        self.archive = Path(archive)
        self.archive.mkdir(exist_ok=True, parents=True)
        
        # Create SimPhotons subdirectory for execution
        self.sim_dir = self.archive / "SimPhotons"
        self.sim_dir.mkdir(exist_ok=True, parents=True)

        with resources.path('G4LumaCam', 'bin') as bin_path:
            self.lumacam_executable = os.path.join(bin_path, "lumacam")

    def _process_output(self, process, output_queue, verbosity):
        """Process the output from the simulation in real-time."""
        # Two patterns for event numbers: one for progress updates and one for final events
        event_pattern = re.compile(r'--> Event (\d+) starts\.')
        final_event_pattern = re.compile(r'Simulating Event: (\d+)')
        cleanup_pattern = re.compile(r'Graphics systems deleted\.')
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            line = line.strip()
            if line:
                # Parse standard event progress
                match = event_pattern.match(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                # Parse final events progress
                match = final_event_pattern.search(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                # Check for simulation cleanup
                if cleanup_pattern.search(line):
                    output_queue.put(('complete', None))
                
                # Handle other output based on verbosity
                if verbosity >= VerbosityLevel.DETAILED:
                    output_queue.put(('output', line))
                elif verbosity >= VerbosityLevel.BASIC and ('starts.' in line or 'Run' in line):
                    output_queue.put(('output', line))

    def run(self, 
            config_or_file: Optional[str | Config] = None, 
            verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        """
        Run the lumacam executable with either a Config object or a macro file.
        
        Args:
            config_or_file: Either a Config object or path to a macro file
            verbosity: Level of output verbosity
            
        Returns:
            pandas.DataFrame: Combined data from all batch files if batching is enabled,
                            otherwise data from single CSV file
        """
        if not os.path.exists(self.lumacam_executable):
            raise FileNotFoundError(f"lumacam executable not found at {self.lumacam_executable}")

        # Handle Config object
        temp_macro = None
        macro_file = None
        num_events = None
        progress_interval = None
        
        if isinstance(config_or_file, Config):
            # Write macro file to SimPhotons directory
            temp_macro = self.sim_dir / "macro.mac"
            macro_file = config_or_file.write(str(temp_macro))
            num_events = config_or_file.num_events
            progress_interval = config_or_file.progress_interval
            
            # Also copy macro file to archive directory
            shutil.copy(str(temp_macro), str(self.archive / "macro.mac"))
        elif isinstance(config_or_file, str):
            if not os.path.exists(config_or_file):
                raise FileNotFoundError(f"Macro file not found at {config_or_file}")
            # Copy macro file to SimPhotons directory
            macro_file = str(self.sim_dir / "macro.mac")
            shutil.copy(config_or_file, macro_file)
            with open(macro_file, 'r') as f:
                content = f.read()
                match = re.search(r'/run/beamOn\s+(\d+)', content)
                if match:
                    num_events = int(match.group(1))
                match = re.search(r'/run/printProgress\s+(\d+)', content)
                if match:
                    progress_interval = int(match.group(1))

        # Change to SimPhotons directory for execution
        original_dir = os.getcwd()
        os.chdir(str(self.sim_dir))

        try:
            process = subprocess.Popen(
                [self.lumacam_executable, "macro.mac"],  # Use local macro.mac in SimPhotons
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            output_queue = queue.Queue()
            output_thread = threading.Thread(
                target=self._process_output,
                args=(process, output_queue, verbosity)
            )
            output_thread.daemon = True
            output_thread.start()

            if num_events is not None:
                pbar = tqdm(total=num_events, desc="Simulating", unit="events")
            else:
                pbar = None

            last_event = 0
            run_completed = False
            last_update = 0

            while process.poll() is None or not output_queue.empty():
                try:
                    msg_type, content = output_queue.get(timeout=0.1)
                    if msg_type == 'progress':
                        current_event = content
                        if pbar is not None and progress_interval:
                            if current_event - last_update >= progress_interval:
                                pbar.n = min(current_event, num_events)
                                pbar.refresh()
                                last_update = current_event
                        last_event = current_event
                    elif msg_type == 'complete':
                        run_completed = True
                        if pbar is not None:
                            pbar.n = num_events
                            pbar.refresh()
                    elif msg_type == 'output':
                        if verbosity >= VerbosityLevel.BASIC:
                            print(content)
                except queue.Empty:
                    continue

            if pbar is not None:
                if run_completed or last_event >= num_events - 1:
                    pbar.n = num_events
                pbar.refresh()
                pbar.close()

            stderr = process.stderr.read()

            if process.returncode != 0:
                raise RuntimeError(f"lumacam execution failed with error:\n{stderr}")
            
            # Handle batch files if csv_batch_size is set
            if isinstance(config_or_file, Config):
                base_name = config_or_file.csv_filename.rsplit('.', 1)[0]
                extension = config_or_file.csv_filename.rsplit('.', 1)[1]
            else:
                base_name = "sim_data"
                extension = "csv"
            
            # Read and combine all CSV files
            dfs = []
            if (isinstance(config_or_file, Config) and config_or_file.csv_batch_size > 0):
                # Get all batch files
                batch_pattern = f"{base_name}_*.{extension}"
                csv_files = sorted(Path().glob(batch_pattern))
            else:
                # Single file case
                csv_files = [Path(f"{base_name}.{extension}")]
            
            # Process CSV files and remove empty ones
            for csv_file in csv_files:
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    # Check if the dataframe is empty (only headers)
                    if df.empty:
                        # Remove the empty CSV file
                        csv_file.unlink()
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"Removed empty CSV file: {csv_file}")
                    else:
                        dfs.append(df)
            
            # Combine all non-empty DataFrames
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
            else:
                raise FileNotFoundError(f"No valid (non-empty) CSV files found in {self.sim_dir}")

            return combined_df

        finally:
            # Change back to original directory
            os.chdir(original_dir)