import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass, field
import json
from typing import Dict, Any, Optional, Union
from multiprocessing import Pool
import glob
import shutil
import subprocess
try:
    from neutron_event_analyzer import Analyse as NEA
    NEA_AVAILABLE = True
except ImportError:
    NEA_AVAILABLE = False


class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything


@dataclass
class Photon2EventConfig:
    """Configuration for the photon2event step."""
    dSpace_px: int = 40
    dTime_s: float = 50e-9
    durationMax_s: float = 500e-9
    dTime_ext: int = 5

    def write(self, output_file: str=".paramsterSettings.json") -> str:
        """
        Write the photon2event configuration to a JSON file.
        
        Args:
            output_file: The path to save the parameters file.
            
        Returns:
            The path to the created JSON file.
        """
        # Structure the parameters as required
        parameters = {
            "photon2event": {
                "dSpace_px": self.dSpace_px,
                "dTime_s": self.dTime_s,
                "durationMax_s": self.durationMax_s,
                "dTime_ext": self.dTime_ext
            }
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Write the JSON file
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)

        return output_file


@dataclass
class BinningConfig:
    """Configuration for a single binning dimension."""
    nBins: int
    resolution: Optional[float] = None
    resolution_s: Optional[float] = None
    resolution_px: Optional[float] = None
    offset: Optional[float] = None
    offset_s: Optional[float] = None
    offset_px: Optional[float] = None

@dataclass
class EventBinningConfig:
    """Configuration for the event binning step."""
    binning_t: Optional[BinningConfig] = None
    binning_x: Optional[BinningConfig] = None
    binning_y: Optional[BinningConfig] = None
    binning_nPhotons: Optional[BinningConfig] = None
    binning_psd: Optional[BinningConfig] = None
    binning_t_relToExtTrigger: Optional[BinningConfig] = None
    
    @classmethod
    def empty(cls) -> 'EventBinningConfig':
        """Create an empty configuration."""
        return cls()
    
    def tof_binning(self) -> 'EventBinningConfig':
        """
        Add time-of-flight binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_t_relToExtTrigger defined
        """
        self.binning_t_relToExtTrigger = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def psd_binning(self) -> 'EventBinningConfig':
        """
        Add pulse shape discrimination binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_psd defined
        """
        self.binning_psd = BinningConfig(
            resolution=1e-6,
            nBins=100,
            offset=0
        )
        return self
    
    def nphotons_binning(self) -> 'EventBinningConfig':
        """
        Add number of photons binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_nPhotons defined
        """
        self.binning_nPhotons = BinningConfig(
            resolution=1,
            nBins=10,
            offset=0
        )
        return self
    
    def time_binning(self) -> 'EventBinningConfig':
        """
        Add time binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_t defined
        """
        self.binning_t = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def spatial_binning(self) -> 'EventBinningConfig':
        """
        Add spatial (x and y) binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_x and binning_y defined
        """
        self.binning_x = BinningConfig(
            resolution_px=32,
            nBins=8,
            offset_px=0
        )
        self.binning_y = BinningConfig(
            resolution_px=32,
            nBins=8,
            offset_px=0
        )
        return self
    
    def write(self, output_file: str=".parameterEvents.json") -> str:
        """
        Write the event binning configuration to a JSON file.
        
        Args:
            output_file: The path to save the parameters file.
        
        Returns:
            The path to the created JSON file.
        """
        # Structure the parameters as required
        parameters = {"bin_events": {}}
        
        # Add only the configurations that are defined
        if self.binning_t:
            parameters["bin_events"]["binning_t"] = self._get_config_dict(self.binning_t, use_s=True)
        if self.binning_x:
            parameters["bin_events"]["binning_x"] = self._get_config_dict(self.binning_x, use_px=True)
        if self.binning_y:
            parameters["bin_events"]["binning_y"] = self._get_config_dict(self.binning_y, use_px=True)
        if self.binning_nPhotons:
            parameters["bin_events"]["binning_nPhotons"] = self._get_config_dict(self.binning_nPhotons)
        if self.binning_psd:
            parameters["bin_events"]["binning_psd"] = self._get_config_dict(self.binning_psd)
        if self.binning_t_relToExtTrigger:
            parameters["bin_events"]["binning_t_relToExtTrigger"] = self._get_config_dict(self.binning_t_relToExtTrigger, use_s=True)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write the JSON file
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        return output_file
    
    def _get_config_dict(self, config: BinningConfig, use_s: bool = False, use_px: bool = False) -> Dict[str, Any]:
        """Convert a BinningConfig to a dictionary with appropriate keys."""
        if config is None:
            return {}
            
        result = {"nBins": config.nBins}
        
        if use_s:
            result["resolution_s"] = config.resolution_s if config.resolution_s is not None else config.resolution
            result["offset_s"] = config.offset_s if config.offset_s is not None else (config.offset or 0)
        elif use_px:
            result["resolution_px"] = config.resolution_px if config.resolution_px is not None else config.resolution
            result["offset_px"] = config.offset_px if config.offset_px is not None else (config.offset or 0)
        else:
            result["resolution"] = config.resolution if config.resolution is not None else 0
            result["offset"] = config.offset if config.offset is not None else 0
        
        return result


class Analysis:
    def __init__(self, archive: str = "test",
                 data: "pd.DataFrame" = None,
                 sim_data: "pd.DataFrame" = None,
                 empir_dirpath: str = None):
        """
        Analysis class for the LumaCam data.

        Inputs:
            archive: str - The directory containing traced photon data.
            data: pd.DataFrame, Optional - The data to be analysed directly.
            sim_data: pd.DataFrame, Optional - The simulation data to be analysed directly.
            empir_dirpath: str, Optional - The path to the empir directory. If None, will use
                           EMPIR_PATH from config if available, otherwise defaults to "./empir".
        
        The class will:
        1. Use the provided DataFrame if `data` is given.
        2. Otherwise, load all traced photon files in `archive/TracedPhotons/`.
        """
        self.archive = Path(archive)
        
        # Determine EMPIR directory path
        if empir_dirpath is not None:
            # Use explicitly provided path
            self.empir_dirpath = Path(empir_dirpath)
        else:
            # Try to load from config
            try:
                from G4LumaCam.config.paths import EMPIR_PATH
                self.empir_dirpath = Path(EMPIR_PATH)
            except ImportError:
                # Fall back to default
                self.empir_dirpath = Path("./empir")
                
        self.traced_dir = self.archive / "TracedPhotons"
        self.sim_dir = self.archive / "SimPhotons"
        self.photon_files_dir = self.archive / "photonFiles"
        self.photon_files_dir.mkdir(parents=True, exist_ok=True)

        self.Photon2EventConfig = Photon2EventConfig
        self.EventBinningConfig = EventBinningConfig

        # Handle data input
        if data is not None:
            self.data = data
        else:
            # Load all traced photon files
            if not self.traced_dir.exists():
                raise FileNotFoundError(f"{self.traced_dir} does not exist.")
            
            traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
            if not traced_files:
                raise FileNotFoundError(f"No traced simulation data found in {self.traced_dir}.")

            valid_dfs = []
            for file in tqdm(traced_files, desc="Loading traced data"):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")

            if valid_dfs:
                self.data = pd.concat(valid_dfs, ignore_index=True)
            else:
                raise ValueError("No valid traced data found!")

        # Handle sim data input
        if sim_data is not None:
            self.sim_data = sim_data
        else:
            # Load all sim photon files
            if not self.sim_dir.exists():
                raise FileNotFoundError(f"{self.sim_dir} does not exist.")
            
            sim_files = sorted(self.sim_dir.glob("sim_data_*.csv"))
            if not sim_files:
                raise FileNotFoundError(f"No sim simulation data found in {self.sim_dir}.")

            valid_dfs = []
            for file in tqdm(sim_files, desc="Loading sim data"):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")

            if valid_dfs:
                self.sim_data = pd.concat(valid_dfs, ignore_index=True)
            else:
                raise ValueError("No valid sim data found!")

        # Validate empir directory and executables
        if not self.empir_dirpath.exists():
            raise FileNotFoundError(f"{self.empir_dirpath} does not exist.")
        
        required_files = {
            "empir_import_photons": "empir_import_photons",
            "empir_bin_photons": "empir_bin_photons",
            "empir_bin_events": "empir_bin_events",
            "process_photon2event": "process_photon2event.sh",
            "empir_export_events": "empir_export_events",
            "empir_export_photons": "empir_export_photons",
            "empir_pixel2photon_tpx3spidr": "bin/empir_pixel2photon_tpx3spidr",
            "empir_photon2event": "bin/empir_photon2event",
            "empir_event2image": "bin/empir_event2image"
        }
        
        self.executables = {}
        for attr_name, filename in required_files.items():
            file_path = self.empir_dirpath / filename
            if not file_path.exists():
                raise FileNotFoundError(f"{filename} not found in {self.empir_dirpath}")
            self.executables[attr_name] = file_path
            setattr(self, attr_name, file_path)

        # Initialize nea-related attributes
        self.events_df = None
        self.photons_df = None
        self.associated_df = None

        # Default parameter settings
        self.default_params = {
            "in_focus": {
                "pixel2photon": {
                    "dSpace": 60,
                    "dTime": 50e-8,
                    "nPxMin": 1,
                    "nPxMax": 1,
                    "TDC1": True
                },
                "photon2event": {
                    "dSpace_px": 40,
                    "dTime_s": 50e-8,
                    "durationMax_s": 500e-8
                },
                "event2image": {
                    "size_x": 512,
                    "size_y": 512,
                    "nPhotons_min": 1,
                    "nPhotons_max": 999,
                    "time_res_s": 1.5625e-9,
                    "time_limit": 640,
                    "psd_min": 0,
                    "time_extTrigger": "reference"
                }
            },
            "out_of_focus": {
                "pixel2photon": {
                    "dSpace": 2,
                    "dTime": 5e-8,
                    "nPxMin": 2,
                    "nPxMax": 999,
                    "TDC1": True
                },
                "photon2event": {
                    "dSpace_px": 40,
                    "dTime_s": 50e-8,
                    "durationMax_s": 500e-8
                },
                "event2image": {
                    "size_x": 512,
                    "size_y": 512,
                    "nPhotons_min": 2,
                    "nPhotons_max": 999,
                    "time_res_s": 1.5625e-9,
                    "time_limit": 640,
                    "psd_min": 0,
                    "time_extTrigger": "reference"
                }
            }
        }

    def _process_single_file(self, file, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Processes a single traced photon file and runs the empir_import_photons script.
        """
        try:
            df = pd.read_csv(file)
            if df.empty:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"⚠️ Skipping empty file: {file.name}")
                return
            
            df = df[["x2", "y2", "toa2"]].dropna()
            df["toa2"] *= 1e-9
            df["px"] = (df["x2"] + 10) / 10 * 128  # convert between px and mm
            df["py"] = (df["y2"] + 10) / 10 * 128  # convert between px and mm
            df = df[["px", "py", "toa2"]]
            df.columns = ["x [px]", "y [px]", "t [s]"]
            df["t_relToExtTrigger [s]"] = df["t [s]"]
            # df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]

            # Create ImportedPhotons directory if it doesn't exist
            imported_photons_dir = self.archive / "ImportedPhotons"
            imported_photons_dir.mkdir(exist_ok=True)

            # Save to ImportedPhotons directory
            output_csv = imported_photons_dir / f"imported_{file.stem}.csv"
            df.sort_values("t [s]").to_csv(output_csv, index=False)

            # Output empirphot file
            empir_file = self.photon_files_dir / f"{file.stem}.empirphot"
            os.system(f"{self.empir_import_photons} {output_csv} {empir_file} csv")
            if verbosity > VerbosityLevel.BASIC:
                print(f"✔ Processed {file.name} → {empir_file.name}")

        except Exception as e:
            if verbosity > VerbosityLevel.BASIC: 
                print(f"❌ Error processing {file.name}: {e}")

    def _run_import_photons(self, parallel=True, clean=True, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Runs empir_import_photons for all traced photon files.
        Args:
            parallel: bool - Whether to process files in parallel using multiprocessing.
            clean: bool - Whether to delete existing .empirphot files before processing.
            verbosity: VerbosityLevel - Controls the level of output during processing.
        """
        traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
        if verbosity > VerbosityLevel.BASIC:
            print(f"Processing {len(traced_files)} traced photon files...")

        if clean:
            existing_empir_files = list(self.photon_files_dir.glob("*.empirphot"))
            for file in existing_empir_files:
                try:
                    file.unlink()
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Could not delete {file.name}: {e}")

        if parallel:
            with Pool() as pool:
                list(tqdm(pool.imap_unordered(self._process_single_file, traced_files), 
                          total=len(traced_files), desc="Processing files"))
        else:
            for file in tqdm(traced_files, desc="Processing files"):
                self._process_single_file(file, verbosity=verbosity)
        if verbosity > VerbosityLevel.BASIC:
            print("✅ Finished processing all files!")

    def _run_photon2event(self, archive: str = None, 
                         config: Photon2EventConfig = None,
                         verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                         **config_kwargs):
        """
        Run the photon2event script with the given configuration, processing all .empirphot files.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        
        # Ensure eventFiles directory exists
        (archive / "eventFiles").mkdir(parents=True, exist_ok=True)
        
        # Get all empirphot files
        input_folder = archive / "photonFiles"
        empirphot_files = list(input_folder.glob("*.empirphot"))
        
        if not empirphot_files:
            raise FileNotFoundError(f"No empirphot files found in {input_folder}")
        
        # Create config file
        params_file = archive / "parameterSettings.json"
        if config is None:
            config = Photon2EventConfig(**config_kwargs)
        config.write(params_file)
        
        # Process each empirphot file
        for empirphot_file in empirphot_files:
            output_file = archive / "eventFiles" / f"{empirphot_file.stem}.empirevent"
            process_command = (
                f"{self.empir_dirpath}/bin/empir_photon2event "
                f"-i '{empirphot_file}' "
                f"-o '{output_file}' "
                f"--paramsFile '{params_file}'"
            )
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running command: {process_command}")
                subprocess.run(process_command, shell=True)
            else:
                with tqdm(total=100, desc=f"Processing {empirphot_file.name}", disable=(verbosity == VerbosityLevel.QUIET)) as pbar:
                    subprocess.run(process_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(10):
                        pbar.update(10)

    def _run_export_events(self, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirevent files from eventFiles subfolder to CSV files in ExportedEvents subfolder,
        with modified headers and type conversions.
        
        Args:
            verbosity: VerbosityLevel - Controls the level of output during processing.
        """
        # Ensure eventFiles directory exists
        event_files_dir = self.archive / "eventFiles"
        if not event_files_dir.exists():
            raise FileNotFoundError(f"{event_files_dir} does not exist.")
        
        # Create ExportedEvents directory
        exported_events_dir = self.archive / "ExportedEvents"
        exported_events_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all .empirevent files
        empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
        if not empirevent_files:
            raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirevent_files)} .empirevent files to CSV...")
        
        # Process each .empirevent file
        for empirevent_file in tqdm(empirevent_files, desc="Exporting events", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                # Define output CSV path
                event_result_csv = exported_events_dir / f"{empirevent_file.stem}.csv"
                
                # Construct command
                cmd = (
                    f"{self.empir_dirpath}/empir_export_events "
                    f"{empirevent_file} "
                    f"{event_result_csv} "
                    f"csv"
                )
                
                # Execute command
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Modify headers and convert types
                try:
                    df = pd.read_csv(event_result_csv)
                    df.columns = ["x", "y", "t", "n", "PSD", "tof"]
                    df["tof"] = df["tof"].astype(float)
                    df["PSD"] = df["PSD"].astype(float)
                    df.to_csv(event_result_csv, index=False)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✔ Exported and modified {empirevent_file.name} → {event_result_csv.name}")
                        
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Error modifying headers for {event_result_csv.name}: {e}")
                    
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"❌ Error exporting {empirevent_file.name}: {e}")
        
        if verbosity > VerbosityLevel.BASIC:
            print("✅ Finished exporting and modifying all event files!")

    def _run_export_photons(self, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirphot files from photonFiles subfolder to CSV files in ImportedPhotons subfolder.
        
        Args:
            verbosity: VerbosityLevel - Controls the level of output during processing.
        """
        # Ensure photonFiles directory exists
        photon_files_dir = self.archive / "photonFiles"
        if not photon_files_dir.exists():
            raise FileNotFoundError(f"{photon_files_dir} does not exist.")
        
        # Ensure ImportedPhotons directory exists
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all .empirphot files
        empirphot_files = sorted(photon_files_dir.glob("*.empirphot"))
        if not empirphot_files:
            raise FileNotFoundError(f"No .empirphot files found in {photon_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirphot_files)} .empirphot files to CSV...")
        
        # Process each .empirphot file
        for empirphot_file in tqdm(empirphot_files, desc="Exporting photons", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                # Define output CSV path
                photon_result_csv = imported_photons_dir / f"imported_{empirphot_file.stem}.csv"
                
                # Construct command
                cmd = (
                    f"{self.empir_dirpath}/empir_export_photons "
                    f"{empirphot_file} "
                    f"{photon_result_csv} "
                    f"csv"
                )
                
                # Execute command
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Modify headers and convert types
                try:
                    df = pd.read_csv(photon_result_csv)
                    df.columns = ["x", "y", "t", "tof"]
                    df["x"] = df["x"].astype(float)
                    df["y"] = df["y"].astype(float)
                    df["t"] = df["t"].astype(float)
                    df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
                    df.to_csv(photon_result_csv, index=False)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✔ Exported and modified {empirphot_file.name} → {photon_result_csv.name}")
                        
                except Exception as e:
                    if verbosity >  VerbosityLevel.BASIC:
                        print(f"⚠️ Error modifying headers for {photon_result_csv.name}: {e}")
                    
            except Exception as e:
                if verbosity >  VerbosityLevel.BASIC:
                    print(f"❌ Error exporting {empirphot_file.name}: {e}")
        
        if verbosity >  VerbosityLevel.BASIC:
            print("✅ Finished exporting and modifying all photon files!")

    def _run_event_binning(self, archive: str = None, 
                          config: EventBinningConfig = None,
                          verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                          **config_kwargs):
        """
        Run the binning script with the given configuration, processing all .empirevent files.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        
        # Ensure eventFiles directory exists
        input_folder = archive / "eventFiles"
        if not input_folder.exists():
            raise FileNotFoundError(f"No eventFiles folder found at {input_folder}")
        
        # Create config file
        params_file = archive / "parameterEvents.json"
        if config is None:
            config = EventBinningConfig(**config_kwargs)
        config.write(params_file)
        
        # Run binning on all empirevent files
        output_file = archive / "binned.empirevent"
        process_command = (
            f"{self.empir_bin_events} "
            f"-I '{input_folder}' "
            f"-o '{output_file}' "
            f"--paramsFile '{params_file}' "
            f"--fileFormat csv"
        )
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Running command: {process_command}")
            subprocess.run(process_command, shell=True)
        else:
            with tqdm(total=100, desc="Running binning", disable=(verbosity == VerbosityLevel.QUIET)) as pbar:
                subprocess.run(process_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(10):
                    pbar.update(10)

    def _read_binned_data(self, archive: str = None) -> pd.DataFrame:
        """
        Read the binned data from the output file.

        Inputs:
            archive: str, The name of the directory to save the data

        Returns:
            The binned data as a DataFrame
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)

        binned_data = pd.read_csv(archive/"binned.empirevent", header=0)
        return binned_data

    def associate_events_photons(self, use_csv: bool = False, time_norm_ns: float = 1.0, 
                               spatial_norm_px: float = 1.0, dSpace_px: float = np.inf,
                               max_time_ns: float = 500, verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                               method: str = 'auto', n_threads: int = 10):
        """
        Associate photons to events using neutron_event_analyzer, either by loading .empirphot and .empirevent
        files or directly from CSV files in ImportedPhotons and ExportedEvents.

        Args:
            use_csv (bool): If True, load data directly from CSV files in ImportedPhotons and ExportedEvents.
                           If False, use neutron_event_analyzer's load method to process .empirphot and .empirevent files.
            time_norm_ns (float): Time normalization factor in nanoseconds for association.
            spatial_norm_px (float): Spatial normalization factor in pixels for association.
            dSpace_px (float): Maximum allowed center-of-mass distance in pixels.
            max_time_ns (float): Maximum time duration in nanoseconds for association.
            verbosity (VerbosityLevel): Level of output verbosity.
            method (str): Association method ('auto', 'kdtree', 'window', 'simple').
            n_threads (int): Number of threads for parallel processing.
        """
        if not NEA_AVAILABLE:
            raise ImportError("neutron_event_analyzer is required for this method.")

        if use_csv:
            # Load from CSV files in ImportedPhotons and ExportedEvents
            imported_photons_dir = self.archive / "ImportedPhotons"
            exported_events_dir = self.archive / "ExportedEvents"
            
            if not imported_photons_dir.exists():
                raise FileNotFoundError(f"{imported_photons_dir} does not exist.")
            if not exported_events_dir.exists():
                raise FileNotFoundError(f"{exported_events_dir} does not exist.")
            
            photon_files = sorted(imported_photons_dir.glob("imported_*.csv"))
            event_files = sorted(exported_events_dir.glob("*.csv"))
            
            if not photon_files:
                raise FileNotFoundError(f"No CSV files found in {imported_photons_dir}")
            if not event_files:
                raise FileNotFoundError(f"No CSV files found in {exported_events_dir}")
            
            if verbosity > VerbosityLevel.BASIC:
                print(f"Loading {len(photon_files)} photon CSV files and {len(event_files)} event CSV files...")
            
            photon_dfs = []
            event_dfs = []
            
            # Load photon CSVs
            for file in tqdm(photon_files, desc="Loading photon CSVs", disable=(verbosity == VerbosityLevel.QUIET)):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        photon_dfs.append(df)
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
            
            # Load event CSVs
            for file in tqdm(event_files, desc="Loading event CSVs", disable=(verbosity == VerbosityLevel.QUIET)):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        event_dfs.append(df)
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
            
            if photon_dfs:
                self.photons_df = pd.concat(photon_dfs, ignore_index=True)
            else:
                self.photons_df = pd.DataFrame()
                if verbosity > VerbosityLevel.BASIC:
                    print("⚠️ No valid photon data loaded from CSV files.")
            
            if event_dfs:
                self.events_df = pd.concat(event_dfs, ignore_index=True)
            else:
                self.events_df = pd.DataFrame()
                if verbosity > VerbosityLevel.BASIC:
                    print("⚠️ No valid event data loaded from CSV files.")
            
        else:
            # Use neutron_event_analyzer to load .empirphot and .empirevent files
            nea = NEA(data_folder=str(self.archive), export_dir=str(self.empir_dirpath), n_threads=n_threads)
            nea.load()
            self.events_df = nea.events_df
            self.photons_df = nea.photons_df
        
        # Perform association
        if not self.events_df.empty and not self.photons_df.empty:
            nea = NEA(data_folder=str(self.archive), export_dir=str(self.empir_dirpath), n_threads=n_threads)
            # Since nea.load() was used or we loaded CSVs, assign pair_dfs manually
            nea.pair_dfs = [(self.events_df, self.photons_df)]
            nea.associate(time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px, dSpace_px=dSpace_px,
                          max_time_ns=max_time_ns, verbosity=verbosity, method=method)
            self.associated_df = nea.get_combined_dataframe()
            
            if verbosity > VerbosityLevel.BASIC:
                print(f"✅ Associated {len(self.associated_df)} photons with events.")
        else:
            self.associated_df = pd.DataFrame()
            if verbosity > VerbosityLevel.BASIC:
                print("⚠️ No data to associate (empty events_df or photons_df).")

    def process_data(self, 
                    dSpace_px: float = 4.0,
                    dTime_s: float = 50e-9,
                    durationMax_s: float = 500e-9,
                    dTime_ext: float = 1.0,
                    nBins: int = 1000,
                    nPhotons_bins: int = None,
                    binning_time_resolution: float = 1.5625e-9,
                    binning_offset: float = 0.0,
                    verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                    suffix: str = "",
                    method: str = 'default',
                    time_norm_ns: float = 1.0,
                    spatial_norm_px: float = 1.0,
                    max_time_ns: float = 500,
                    n_threads: int = 10) -> pd.DataFrame:
        """
        Streamlined method to process data through the complete analysis pipeline.
        
        Args:
            dSpace_px: Spatial clustering distance in pixels
            dTime_s: Time clustering threshold in seconds
            durationMax_s: Maximum event duration in seconds
            dTime_ext: Time extension factor for clustering
            nBins: Number of time bins
            nPhotons_bins: Number of photon bins
            binning_time_resolution: Time resolution for binning in seconds
            binning_offset: Time offset for binning in seconds
            verbosity: Level of output verbosity
            suffix: Optional suffix for output folder and files
            method: Processing method ('default' or 'nea' for neutron_event_analyzer)
            time_norm_ns: Time normalization factor in nanoseconds for nea association
            spatial_norm_px: Spatial normalization factor in pixels for nea association
            max_time_ns: Maximum time duration in nanoseconds for nea association
            n_threads: Number of threads for parallel processing in nea
        
        Returns:
            DataFrame with processed data
        """
        # Create base directories
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ImportedPhotons directory
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create suffixed subfolder for binned results
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup photon2event config
        p2e_config = self.Photon2EventConfig()
        p2e_config.dSpace_px = dSpace_px
        p2e_config.dTime_s = dTime_s
        p2e_config.durationMax_s = durationMax_s
        p2e_config.dTime_ext = dTime_ext
        params_file = suffix_dir / "parameterSettings.json"
        p2e_config.write(params_file)
        
        # Run import photons and photon2event
        self._run_import_photons(verbosity=verbosity)
        self._run_photon2event(config=p2e_config, verbosity=verbosity)
        self._run_export_events(verbosity=verbosity)
        
        if method == 'nea':
            # Create AssociatedResults directory
            associated_dir = self.archive / "AssociatedResults"
            associated_dir.mkdir(parents=True, exist_ok=True)
            
            # Create prefix subfolder (using suffix as prefix)
            prefix_dir = associated_dir / (suffix.strip("_") if suffix else "default")
            prefix_dir.mkdir(parents=True, exist_ok=True)
            
            # Export photons for nea compatibility
            self._run_export_photons(verbosity=verbosity)
            
            # Run nea association
            self.associate_events_photons(use_csv=True, time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px,
                                        dSpace_px=dSpace_px, max_time_ns=max_time_ns, verbosity=verbosity,
                                        method='kdtree', n_threads=n_threads)
            
            # Save associated dataframe
            if not self.associated_df.empty:
                output_csv = prefix_dir / "associated_results.csv"
                self.associated_df.to_csv(output_csv, index=False)
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Associated data saved to {output_csv}")
        
        # Setup event binning config
        binning_config = self.EventBinningConfig().time_binning()
        binning_config.binning_t.nBins = nBins
        binning_config.binning_t.resolution_s = binning_time_resolution
        binning_config.binning_t.offset_s = binning_offset
        if nPhotons_bins is not None:
            binning_config = binning_config.nphotons_binning()
            binning_config.binning_nPhotons.nBins = nPhotons_bins
        event_params_file = suffix_dir / "parameterEvents.json"
        binning_config.write(event_params_file)
        
        # Run event binning
        self._run_event_binning(config=binning_config, verbosity=verbosity)
        
        # Read and process binned data
        result_df = self._read_binned_data()
        if nPhotons_bins is None:
            result_df.columns = ["stacks", "counts"]
        else:
            result_df.columns = ["stacks", "nPhotons", "counts"]
        result_df["err"] = np.sqrt(result_df["counts"])
        result_df["stacks"] = np.arange(len(result_df))
        
        # Save binned results in suffixed folder
        output_csv = suffix_dir / "counts.csv"
        result_df.to_csv(output_csv, index=False)
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Processed data saved to {output_csv}")
        
        return result_df

    def process_data_event_by_event(self,
                                   dSpace_px: float = 4.0,
                                   dTime_s: float = 50e-9,
                                   durationMax_s: float = 500e-9,
                                   dTime_ext: float = 1.0,
                                   nBins: int = 1000,
                                   binning_time_resolution: float = 1.5625e-9,
                                   binning_offset: float = 0.0,
                                   verbosity: int = 0,
                                   merge: bool = False,
                                   suffix: str = "",
                                   time_norm_ns: float = 1.0,
                                   spatial_norm_px: float = 1.0,
                                   fov: float = 120.0,
                                   focus_factor: float = 1.2,
                                   deadtime: float = None) -> pd.DataFrame:
        """
        Processes data event by event, grouping optical photons by neutron_id.
        
        This method:
        1. Combines self.data with neutron_id from self.sim_data using a merge
        2. Determines the number of batches from SimPhotons or TracedPhotons folder
        3. Groups traced photons by neutron_id and organizes them by batch
        4. Processes each neutron event independently
        5. Combines results by batch into separate files in a suffixed subfolder
        6. Optionally merges results with simulation and traced data, adding event_id
        7. Saves processed photon CSV files in ImportedPhotons directory
        
        Args:
            dSpace_px: Spatial clustering distance in pixels
            dTime_s: Time clustering threshold in seconds
            durationMax_s: Maximum event duration in seconds
            dTime_ext: Time extension factor for clustering
            nBins: Number of time bins
            binning_time_resolution: Time resolution for binning in seconds
            binning_offset: Time offset for binning in seconds
            verbosity: Level of output verbosity
            merge: If True, merge results with simulation and traced data and save
            suffix: Optional suffix for output folder and files
            time_norm_ns: Normalization factor for time differences (ns) in matching
            spatial_norm_px: Normalization factor for spatial differences (px) in matching
            fov: Field of view in mm
            focus_factor: Factor that relates the hit position on the sensor to the actual hit position on the scintillator screen
            deadtime: Deadtime in nanoseconds for photon grouping (optional)
        
        Returns:
            DataFrame with processed event data (optionally merged with sim_data and traced_data)
        """
        # Create base directories
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ImportedPhotons directory
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create suffixed subfolder
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directory for event processing
        temp_dir = self.archive / "temp_events"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for batch results
        result_dir = self.archive / "EventResults"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of batches
        sim_photons_path = self.archive / "SimPhotons"
        traced_photons_path = self.archive / "TracedPhotons"  
        
        if sim_photons_path.exists():
            batch_files = list(sim_photons_path.glob("*.csv"))
            num_batches = len(batch_files)
            if verbosity >= 1:
                print(f"Detected {num_batches} batches from SimPhotons folder")
        elif traced_photons_path.exists():
            batch_files = list(traced_photons_path.glob("*.csv"))
            num_batches = len(batch_files)
            if verbosity >= 1:
                print(f"Detected {num_batches} batches from TracedPhotons folder")
        else:
            num_batches = 1
            if verbosity >= 1:
                print("No batch information found, defaulting to a single batch")
        
        # Combine data with simulation data
        if verbosity >= 1:
            print("Combining data with simulation data to include neutron_id...")
        
        if 'neutron_id' not in self.sim_data.columns:
            raise ValueError("Simulation data must contain a 'neutron_id' column")
        
        # Merge self.data with self.sim_data based on neutron_id
        combined_data = self.sim_data.copy()
        if not self.data.empty:
            required_cols = ['neutron_id', 'x2', 'y2', 'z2', 'toa2']
            if deadtime is not None:
                required_cols += ['photon_count', 'time_diff', 'nz', 'pz']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                if verbosity >= 1:
                    print(f"Cannot merge data: missing columns {missing_cols} in traced data")
                for col in ['x2', 'y2', 'z2', 'toa2', 'photon_count', 'time_diff', 'nz', 'pz']:
                    combined_data[col] = np.nan
            else:
                # Create a mapping from neutron_id to traced data
                traced_cols = ['x2', 'y2', 'z2', 'toa2']
                if deadtime is not None:
                    traced_cols += ['photon_count', 'time_diff', 'nz', 'pz']
                neutron_id_map = self.data.groupby('neutron_id')[traced_cols].first().reset_index()
                combined_data = combined_data.merge(neutron_id_map, on='neutron_id', how='left')
        
        # Apply spatial normalization if fov is provided
        if fov is not None:
            combined_data['x2'] = combined_data['x2'] / fov * 256
            combined_data['y2'] = combined_data['y2'] / fov * 256
        
        has_batch_id = 'batch_id' in self.sim_data.columns
        
        if has_batch_id:
            combined_data['batch_id'] = self.sim_data['batch_id']
            if verbosity >= 1:
                print("Using batch_id from simulation data")
        else:
            unique_neutron_ids = combined_data['neutron_id'].unique()
            events_per_batch = len(unique_neutron_ids) // num_batches + 1
            neutron_to_batch = {nid: min(i // events_per_batch, num_batches - 1) 
                                for i, nid in enumerate(unique_neutron_ids)}
            combined_data['batch_id'] = combined_data['neutron_id'].map(neutron_to_batch)
            if verbosity >= 1:
                print(f"Assigned {len(unique_neutron_ids)} neutron events to {num_batches} batches")
        
        # Group data by neutron_id
        if verbosity >= 1:
            print("Grouping data by neutron_id...")
        
        neutron_groups = combined_data.groupby('neutron_id')
        
        batch_results = {i: [] for i in range(num_batches)}
        
        # Setup photon2event config
        p2e_config = self.Photon2EventConfig()
        p2e_config.dSpace_px = dSpace_px
        p2e_config.dTime_s = dTime_s
        p2e_config.durationMax_s = durationMax_s
        p2e_config.dTime_ext = dTime_ext
        params_file = suffix_dir / "parameterSettings.json"
        p2e_config.write(params_file)
        
        # Setup event binning config
        binning_config = self.EventBinningConfig().time_binning()
        binning_config.binning_t.nBins = nBins
        binning_config.binning_t.resolution_s = binning_time_resolution
        binning_config.binning_t.offset_s = binning_offset
        event_params_file = suffix_dir / "parameterEvents.json"
        binning_config.write(event_params_file)
        
        # Process each neutron event
        for neutron_id, group in tqdm(neutron_groups, desc="Processing neutron events"):
            try:
                if group.empty:
                    continue
                    
                batch_id = group['batch_id'].iloc[0]
                
                df = group[["x2", "y2", "toa2"]].dropna()
                df["toa2"] *= 1e-9  # Convert toa2 from ns to s for processing
                df["px"] = (df["x2"] + 10) / 10 * 128
                df["py"] = (df["y2"] + 10) / 10 * 128
                df = df[["px", "py", "toa2"]]
                df.columns = ["x [px]", "y [px]", "t [s]"]
                df["t_relToExtTrigger [s]"] = df["t [s]"]
                df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
                
                if df.empty:
                    continue
                
                event_prefix = f"event_{neutron_id}"
                # Save to ImportedPhotons directory
                imported_csv = imported_photons_dir / f"imported_{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(imported_csv, index=False)
                
                # Still use temp_dir for intermediate files
                temp_csv = temp_dir / f"{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(temp_csv, index=False)
                
                empirphot_file = temp_dir / f"{event_prefix}.empirphot"
                cmd = f"{self.empir_import_photons} {temp_csv} {empirphot_file} csv"
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                empirevent_file = temp_dir / f"{event_prefix}.empirevent"
                cmd = (
                    f"{self.empir_dirpath}/bin/empir_photon2event "
                    f"-i '{empirphot_file}' "
                    f"-o '{empirevent_file}' "
                    f"--paramsFile '{params_file}'"
                )
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                event_result_csv = temp_dir / f"{event_prefix}_result.csv"
                cmd = (
                    f"{self.empir_dirpath}/empir_export_events "
                    f"{empirevent_file} "
                    f"{event_result_csv} "
                    f"csv"
                )
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if event_result_csv.exists():
                    try:
                        result_df = pd.read_csv(event_result_csv)
                        result_df.columns = [col.strip() for col in result_df.columns]
                        expected_columns = ['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value',
                                            't_relToExtTrigger [s]']
                        for col in expected_columns:
                            if col not in result_df.columns:
                                raise ValueError(f"Missing expected column '{col}' in result CSV")
                        result_df['neutron_id'] = neutron_id
                        batch_results[batch_id].append(result_df)
                    except pd.errors.EmptyDataError:
                        if verbosity >= 1:
                            print(f"Empty result for neutron_id {neutron_id}")
                
                # Clean up temporary files (but keep imported_csv)
                if verbosity < 2:
                    temp_csv.unlink(missing_ok=True)
                    empirphot_file.unlink(missing_ok=True)
                    empirevent_file.unlink(missing_ok=True)
                    event_result_csv.unlink(missing_ok=True)
                    
            except Exception as e:
                if verbosity >= 1:
                    print(f"Error processing neutron_id {neutron_id}: {str(e)}")
        
        # Combine and save results by batch
        all_results = []
        for batch_id, results in batch_results.items():
            if not results:
                if verbosity >= 1:
                    print(f"No valid results for batch {batch_id}")
                continue
                
            batch_df = pd.concat(results, ignore_index=True)
            all_results.append(batch_df)
            
            batch_csv = result_dir / f"batch_{batch_id}_results.csv"
            batch_df.to_csv(batch_csv, index=False)
            
            if verbosity >= 1:
                print(f"Batch {batch_id} results saved to {batch_csv} ({len(results)} neutron events)")
        
        if not all_results:
            if verbosity >= 1:
                print("No valid results found!")
            return pd.DataFrame()
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results in suffixed folder
        combined_csv = suffix_dir / "all_batches_results.csv"
        combined_results.to_csv(combined_csv, index=False)
        
        if verbosity >= 1:
            print(f"All results saved to {combined_csv}")
            print(f"Processed {len(combined_results)} neutron events across {len(batch_results)} batches")
        
        # Optional merging with simulation and traced data
        if merge:
            if verbosity >= 1:
                print("Merging processed results with simulation and traced data...")
            
            def merge_sim_and_recon_data(sim_data, traced_data, recon_data, fov: float = 120, focus_factor: float = 1.2):
                """
                Merge simulation, traced photon, and reconstruction dataframes based on neutron_id.
                Matches reconstructed events to simulation photons by minimizing a combined
                time and spatial distance metric.
                Adds event_id for each reconstructed event per neutron_id.
                
                Args:
                    sim_data: DataFrame with simulation data
                    traced_data: DataFrame with traced photon data
                    recon_data: DataFrame with reconstructed event data
                    fov: Field of View in mm, used to scale photon positions
                    focus_factor: Factor for hit position on sensor to scintillator screen
                
                Returns:
                    Merged DataFrame with simulation, traced, and reconstruction columns
                """
                sim_df = sim_data.copy()
                traced_df = traced_data.copy()
                recon_df = recon_data.copy()
                
                # Initialize traced columns in sim_df
                sim_df['x2'] = np.nan
                sim_df['y2'] = np.nan
                sim_df['z2'] = np.nan
                sim_df['toa2'] = np.nan
                sim_df['photon_count'] = np.nan
                sim_df['time_diff'] = np.nan
                sim_df['nz'] = np.nan
                sim_df['pz'] = np.nan
                sim_df['photon_px'] = np.nan
                sim_df['photon_py'] = np.nan
                
                # Assign traced data columns (based on neutron_id)
                if not traced_df.empty:
                    # Check for standard column names
                    x_col = next((col for col in ['x2', 'x', 'px'] if col in traced_df.columns), None)
                    y_col = next((col for col in ['y2', 'y', 'py'] if col in traced_df.columns), None)
                    z_col = next((col for col in ['z2', 'z'] if col in traced_df.columns), None)
                    time_col = next((col for col in ['toa2', 'time', 'toa'] if col in traced_df.columns), None)
                    
                    required_cols = ['neutron_id']
                    if x_col: required_cols.append(x_col)
                    if y_col: required_cols.append(y_col)
                    if z_col: required_cols.append(z_col)
                    if time_col: required_cols.append(time_col)
                    if deadtime is not None:
                        required_cols += ['photon_count', 'time_diff', 'nz', 'pz']
                    
                    missing_cols = [col for col in required_cols if col not in traced_df.columns]
                    if missing_cols:
                        if verbosity >= 1:
                            print(f"Warning: traced_data missing required columns {missing_cols}. Traced columns remain NaN.")
                    else:
                        # Rename columns to standard names for consistency
                        rename_dict = {}
                        if x_col and x_col != 'x2': rename_dict[x_col] = 'x2'
                        if y_col and y_col != 'y2': rename_dict[y_col] = 'y2'
                        if z_col and z_col != 'z2': rename_dict[z_col] = 'z2'
                        if time_col and time_col != 'toa2': rename_dict[time_col] = 'toa2'
                        temp_traced_df = traced_df.rename(columns=rename_dict)
                        
                        traced_cols = ['x2', 'y2', 'z2', 'toa2']
                        if deadtime is not None:
                            traced_cols += ['photon_count', 'time_diff', 'nz', 'pz']
                        neutron_id_map = temp_traced_df.groupby('neutron_id')[traced_cols].first().reset_index()
                        if verbosity >= 2:
                            print(f"neutron_id_map columns: {neutron_id_map.columns.tolist()}")
                            print(f"neutron_id_map non-NaN counts: {neutron_id_map.notna().sum().to_dict()}")
                            print(f"neutron_id overlap: {len(set(sim_df['neutron_id']).intersection(set(neutron_id_map['neutron_id'])))} common IDs")
                        if not neutron_id_map.empty:
                            sim_df = sim_df.merge(neutron_id_map, on='neutron_id', how='left', suffixes=('', '_traced'))
                            # Handle duplicate columns from merge
                            for col in traced_cols:
                                if f"{col}_traced" in sim_df.columns:
                                    sim_df[col] = sim_df[f"{col}_traced"].combine_first(sim_df[col])
                                    sim_df = sim_df.drop(columns=f"{col}_traced")
                        if verbosity >= 2:
                            print(f"sim_df columns after merge: {sim_df.columns.tolist()}")
                        # Compute photon_px and photon_py only where x2 and y2 are not NaN
                        sim_df['photon_px'] = np.where(sim_df['x2'].notna(), (sim_df['x2'] + 10) / 10 * 128, np.nan)
                        sim_df['photon_py'] = np.where(sim_df['y2'].notna(), (sim_df['y2'] + 10) / 10 * 128, np.nan)
                        if verbosity >= 2:
                            print(f"Assigned traced columns. Non-NaN counts: x2={sim_df['x2'].notna().sum()}, toa2={sim_df['toa2'].notna().sum()}")
                
                # Initialize merged DataFrame with sim_df
                merged_df = sim_df.copy()
                
                # Add reconstruction columns with NaN
                for col in recon_df.columns:
                    if col != 'neutron_id':
                        merged_df[col] = np.nan
                merged_df['event_id'] = np.nan
                merged_df['time_diff_ns'] = np.nan
                merged_df['spatial_diff_px'] = np.nan
                
                # Group reconstruction data by neutron_id
                recon_groups = recon_df.groupby('neutron_id')
                
                # Match reconstruction events to simulation rows
                for neutron_id in sorted(merged_df['neutron_id'].unique()):
                    sim_group = merged_df[merged_df['neutron_id'] == neutron_id].copy()
                    recon_group = recon_groups.get_group(neutron_id) if neutron_id in recon_groups.groups else pd.DataFrame()
                    
                    if recon_group.empty:
                        continue
                    
                    # Assign event_id to reconstruction events
                    recon_group = recon_group.sort_values('t [s]').reset_index(drop=True)
                    recon_group['event_id'] = recon_group.index + 1
                    
                    # For each reconstruction event, find the closest simulation photon(s)
                    for _, recon_row in recon_group.iterrows():
                        n_photons = int(recon_row['nPhotons [1]'])
                        recon_time_s = recon_row['t [s]']
                        recon_x = recon_row['x [px]']
                        recon_y = recon_row['y [px]']
                        event_id = recon_row['event_id']
                        
                        # Compute distances
                        sim_times = sim_group['toa2'].values * 1e-9  # Convert to seconds
                        sim_px = sim_group['photon_px'].values
                        sim_py = sim_group['photon_py'].values
                        
                        time_diffs = np.abs(sim_times - recon_time_s) * 1e9  # Convert to ns
                        spatial_diffs = np.sqrt((sim_px - recon_x)**2 + (sim_py - recon_y)**2)
                        
                        if np.all(np.isnan(sim_px)) or np.all(np.isnan(sim_py)) or np.all(np.isnan(sim_times)):
                            combined_diffs = time_diffs / time_norm_ns
                            spatial_diffs = np.array([np.nan] * len(time_diffs))
                        else:
                            combined_diffs = (time_diffs / time_norm_ns) + (spatial_diffs / spatial_norm_px)
                        
                        # Select closest photon(s)
                        if n_photons == 1:
                            if len(sim_group) > 0:
                                closest_idx = np.argmin(combined_diffs)
                                sim_idx = sim_group.index[closest_idx]
                                for col in recon_df.columns:
                                    if col != 'neutron_id':
                                        merged_df.loc[sim_idx, col] = recon_row[col]
                                merged_df.loc[sim_idx, 'event_id'] = event_id
                                merged_df.loc[sim_idx, 'time_diff_ns'] = time_diffs[closest_idx]
                                merged_df.loc[sim_idx, 'spatial_diff_px'] = spatial_diffs[closest_idx]
                        else:
                            if len(sim_group) >= n_photons:
                                closest_indices = np.argsort(combined_diffs)[:n_photons]
                                selected_px = sim_group.iloc[closest_indices]['photon_px']
                                selected_py = sim_group.iloc[closest_indices]['photon_py']
                                if not (np.all(np.isnan(selected_px)) or np.all(np.isnan(selected_py))):
                                    com_x = selected_px.mean()
                                    com_y = selected_py.mean()
                                    com_dist = np.sqrt((com_x - recon_x)**2 + (com_y - recon_y)**2)
                                    if com_dist > dSpace_px:
                                        continue  # Skip if center of mass is too far
                                for idx in closest_indices:
                                    sim_idx = sim_group.index[idx]
                                    for col in recon_df.columns:
                                        if col != 'neutron_id':
                                            merged_df.loc[sim_idx, col] = recon_row[col]
                                    merged_df.loc[sim_idx, 'event_id'] = event_id
                                    merged_df.loc[sim_idx, 'time_diff_ns'] = time_diffs[idx]
                                    merged_df.loc[sim_idx, 'spatial_diff_px'] = spatial_diffs[idx]
                
                merged_df = self.calculate_reconstruction_stats(merged_df, fov=fov, focus_factor=focus_factor)
                
                # Ensure column order
                sim_cols = [col for col in sim_df.columns if col not in ['x2', 'y2', 'z2', 'toa2', 'photon_count', 'time_diff', 'nz', 'pz', 'photon_px', 'photon_py']]
                traced_cols = ['x2', 'y2', 'z2', 'toa2', 'photon_count', 'time_diff', 'nz', 'pz', 'photon_px', 'photon_py']
                recon_cols = [col for col in recon_df.columns if col != 'neutron_id']
                final_cols = sim_cols + traced_cols + recon_cols + ['event_id', 'time_diff_ns', 'spatial_diff_px', 'x3', 'y3', 'delta_x', 'delta_y', 'delta_r']
                merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]]
                
                return merged_df
            
            # Load traced data
            traced_data = pd.DataFrame()
            if traced_photons_path.exists():
                traced_files = list(traced_photons_path.glob("*.csv"))
                if traced_files:
                    traced_dfs = [pd.read_csv(f) for f in traced_files]
                    traced_data = pd.concat(traced_dfs, ignore_index=True)
                    if verbosity >= 1:
                        print(f"Loaded traced data with columns: {list(traced_data.columns)}")
                else:
                    if verbosity >= 1:
                        print("Warning: No traced photon CSV files found in TracedPhotons folder.")
            
            merged_df = merge_sim_and_recon_data(self.sim_data, traced_data, combined_results, fov=fov, focus_factor=focus_factor)
            
            # Save merged results in suffixed folder
            merged_csv = suffix_dir / "merged_all_batches_results.csv"
            merged_df.to_csv(merged_csv, index=False)
            
            if verbosity >= 1:
                print(f"Merged results saved to {merged_csv}")
            
            return merged_df
        
        return combined_results


    def calculate_reconstruction_stats(self, df, fov=60.0, focus_factor=1.25):
        """
        Calculate reconstructed coordinates and differences for the DataFrame.

        Computes x3, y3 from pixel coordinates (x [px], y [px]) using the provided
        field of view and focus factor, and calculates differences (delta_x, delta_y, delta_r)
        relative to simulation coordinates (x, y).

        Args:
            df: DataFrame containing 'x [px]', 'y [px]' (reconstructed) and 'x', 'y' (simulation)
            fov: Field of view in mm (default: 60.0)
            focus_factor: Factor to compensate for sensor-to-scintillator scaling (default: 1.25)

        Returns:
            DataFrame with added columns x3, y3, delta_x, delta_y, delta_r
        """
        df['x3'] = np.where(df['x [px]'].notna(), -(df['x [px]'] - 128) * fov / 128 * focus_factor, np.nan)
        df['y3'] = np.where(df['y [px]'].notna(), -(df['y [px]'] - 128) * fov / 128 * focus_factor, np.nan)
        df['delta_x'] = np.where(df['x3'].notna() & df['nx'].notna(), df['x3'] - df['nx'], np.nan)
        df['delta_y'] = np.where(df['y3'].notna() & df['ny'].notna(), df['y3'] - df['ny'], np.nan)
        df['delta_r'] = np.where(df['delta_x'].notna() & df['delta_y'].notna(), 
                                np.sqrt(df['delta_x']**2 + df['delta_y']**2), np.nan)
        return df

    def process(self, 
                    params: Union[str, Dict[str, Any]] = None,
                    n_threads: int = 1,
                    suffix: str = "",
                    pixel2photon: bool = True,
                    photon2event: bool = True,
                    event2image: bool = False,
                    export_photons: bool = True,
                    export_events: bool = False,
                    verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                    clean: bool = True,
                    **kwargs) -> None:
            """
            Process TPX3 files through the EMPIR pipeline to produce photonFiles and eventFiles.
            
            Args:
                params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
                    containing the parameters for pixel2photon, photon2event, and event2image.
                    If None, uses default parameters based on focus mode.
                n_threads: Number of threads to use for parallel processing of files.
                suffix: Optional suffix to create a subfolder and symlink TPX3 files for processing.
                pixel2photon: If True, runs empir_import_photons on TPX3 files to generate .empirphot files.
                photon2event: If True, runs empir_photon2event on generated .empirphot files.
                event2image: If True, runs empir_event2image on generated .empirevent files.
                export_photons: If True, runs empir_export_photons on generated .empirphot files.
                export_events: If True, runs empir_export_events on generated .empirevent files.
                verbosity: Controls the level of output (QUIET=0, BASIC=1, DETAILED=2).
                clean: If True, deletes existing .empirphot and .empirevent files before processing.
                **kwargs: Additional parameters to update specific fields in the configuration
                        (e.g., dTime_s, dSpace_px for photon2event).
            """
            import time
            start_time = time.time()

            # Set up processing directory
            base_dir = self.archive
            if suffix:
                process_dir = base_dir / suffix.strip("_")
                process_dir.mkdir(parents=True, exist_ok=True)
                tpx3_dir = process_dir / "tpx3Files"
                tpx3_dir.mkdir(parents=True, exist_ok=True)
                
                # Symlink TPX3 files by changing to tpx3_dir and creating relative links
                orig_tpx3_dir = base_dir / "tpx3Files"
                if not orig_tpx3_dir.exists():
                    raise FileNotFoundError(f"Original tpx3Files directory not found at {orig_tpx3_dir}")
                
                current_dir = os.getcwd()
                try:
                    os.chdir(tpx3_dir)
                    for tpx3_file in orig_tpx3_dir.glob("*.tpx3"):
                        dest_file = tpx3_file.name
                        rel_path = os.path.relpath(tpx3_file, tpx3_dir)
                        if not os.path.exists(dest_file):
                            os.symlink(rel_path, dest_file)
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Created symlink: {tpx3_dir / dest_file} -> {rel_path}")
                finally:
                    os.chdir(current_dir)
            else:
                process_dir = base_dir
                tpx3_dir = base_dir / "tpx3Files"

            # Ensure output directories exist
            photon_files_dir = process_dir / "photonFiles"
            event_files_dir = process_dir / "eventFiles"
            final_dir = process_dir / "final"
            photon_files_dir.mkdir(parents=True, exist_ok=True)
            event_files_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)

            # Clean existing files if requested
            if clean:
                for file in photon_files_dir.glob("*.empirphot"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
                for file in event_files_dir.glob("*.empirevent"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
                for file in final_dir.glob("*"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")

            # Load or set parameters
            if params is None:
                # Use default parameters (in_focus or out_of_focus based on context)
                parameters = self.default_params.get("in_focus", {})
            elif isinstance(params, str):
                if params in ["in_focus", "out_of_focus"]:
                    parameters = self.default_params.get(params, {})
                elif params.endswith('.json'):
                    if not os.path.exists(params):
                        raise FileNotFoundError(f"Parameter file {params} not found")
                    with open(params, 'r') as f:
                        parameters = json.load(f)
                else:
                    try:
                        parameters = json.loads(params)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON string provided for parameters")
            elif isinstance(params, dict):
                parameters = params
            else:
                raise ValueError("params must be 'in_focus', 'out_of_focus', a JSON file path, JSON string, or dictionary")

            # Update parameters with kwargs (e.g., for photon2event settings)
            if kwargs:
                if "pixel2photon" in parameters:
                    parameters["pixel2photon"].update({k: v for k, v in kwargs.items() if k in ["dSpace", "dTime", "nPxMin", "nPxMax", "TDC1"]})
                if "photon2event" in parameters:
                    parameters["photon2event"].update({k: v for k, v in kwargs.items() if k in ["dSpace_px", "dTime_s", "durationMax_s", "dTime_ext"]})
                if "event2image" in parameters:
                    parameters["event2image"].update({k: v for k, v in kwargs.items() if k in ["size_x", "size_y", "nPhotons_min", "nPhotons_max", "time_res_s", "time_limit", "psd_min", "time_extTrigger"]})

            # Write parameters to hidden file
            params_file = process_dir / ".parameterSettings.json"
            with open(params_file, 'w') as f:
                json.dump(parameters, f, indent=4)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Parameters written to {params_file}")

            if pixel2photon:
                if verbosity > VerbosityLevel.BASIC:
                    print("Starting pixel2photon processing...")
                # Process pixel2photon
                tpx3_files = sorted(tpx3_dir.glob("*.tpx3"))
                if not tpx3_files:
                    raise FileNotFoundError(f"No .tpx3 files found in {tpx3_dir}")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Processing {len(tpx3_files)} .tpx3 files to photon files...")

                pids = []
                file_cnt = 0
                result_all = 0

                for tpx3_file in tqdm(tpx3_files, desc="Processing pixel2photon", disable=(verbosity == VerbosityLevel.QUIET)):
                    file_base = tpx3_file.stem
                    file_cnt += 1
                    output_file = photon_files_dir / f"{file_base}.empirphot"
                    cmd = [
                        str(self.empir_dirpath / "bin/empir_pixel2photon_tpx3spidr"),
                        "-i", str(tpx3_file),
                        "-o", str(output_file),
                        "--paramsFile", str(params_file)
                    ]
                    if parameters.get("pixel2photon", {}).get("TDC1", False):
                        cmd.append("-T")
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Running: {' '.join(cmd)}")
                        process = subprocess.Popen(cmd)
                    else:
                        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    pids.append(process)
                    
                    if len(pids) >= n_threads:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Reached {n_threads} files, waiting for processing...")
                        for process in pids:
                            result = process.wait()
                            if result != 0:
                                result_all = 1
                                if verbosity > VerbosityLevel.BASIC:
                                    print(f"Error occurred while processing a file!")
                        pids = []
                        if verbosity > VerbosityLevel.BASIC:
                            print("Processed files, continuing...")

                # Wait for remaining processes
                for process in pids:
                    result = process.wait()
                    if result != 0:
                        result_all = 1
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error occurred while processing a file!")

                if result_all != 0:
                    raise RuntimeError("Errors occurred during pixel2photon processing")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Finished processing {file_cnt} .tpx3 files")

            if photon2event:
                if verbosity > VerbosityLevel.BASIC:
                    print("Starting photon2event processing...")
                # Process photon2event
                empirphot_files = sorted(photon_files_dir.glob("*.empirphot"))
                if not empirphot_files:
                    raise FileNotFoundError(f"No .empirphot files found in {photon_files_dir}")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Processing {len(empirphot_files)} .empirphot files to event files...")

                pids = []
                file_cnt = 0
                result_all = 0

                for empirphot_file in tqdm(empirphot_files, desc="Processing photon2event", disable=(verbosity == VerbosityLevel.QUIET)):
                    file_base = empirphot_file.stem
                    file_cnt += 1
                    output_file = event_files_dir / f"{file_base}.empirevent"
                    cmd = [
                        str(self.empir_dirpath / "bin/empir_photon2event"),
                        "-i", str(empirphot_file),
                        "-o", str(output_file),
                        "--paramsFile", str(params_file)
                    ]
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Running: {' '.join(cmd)}")
                        process = subprocess.Popen(cmd)
                    else:
                        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    pids.append(process)
                    
                    if len(pids) >= n_threads:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Reached {n_threads} files, waiting for processing...")
                        for process in pids:
                            result = process.wait()
                            if result != 0:
                                result_all = 1
                                if verbosity > VerbosityLevel.BASIC:
                                    print(f"Error occurred while processing a file!")
                        pids = []
                        if verbosity > VerbosityLevel.BASIC:
                            print("Processed files, continuing...")

                # Wait for remaining processes
                for process in pids:
                    result = process.wait()
                    if result != 0:
                        result_all = 1
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error occurred while processing a file!")

                if result_all != 0:
                    raise RuntimeError("Errors occurred during photon2event processing")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Finished processing {file_cnt} .empirphot files")

            # Export photons if requested
            if export_photons:
                self._run_export_photons(verbosity=verbosity)

            # Export events if requested
            if export_events:
                self._run_export_events(verbosity=verbosity)

            if event2image:
                if verbosity > VerbosityLevel.BASIC:
                    print("Starting event2image processing...")
                # Process event2image
                empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
                if not empirevent_files:
                    raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Processing {len(empirevent_files)} .empirevent files to final images...")

                pids = []
                file_cnt = 0
                result_all = 0

                for empirevent_file in tqdm(empirevent_files, desc="Processing event2image", disable=(verbosity == VerbosityLevel.QUIET)):
                    file_base = empirevent_file.stem
                    file_cnt += 1
                    output_file = final_dir / f"{file_base}.empirimage"
                    cmd = [
                        str(self.empir_dirpath / "bin/empir_event2image"),
                        "-i", str(empirevent_file),
                        "-o", str(output_file),
                        "--paramsFile", str(params_file)
                    ]
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Running: {' '.join(cmd)}")
                        process = subprocess.Popen(cmd)
                    else:
                        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    pids.append(process)
                    
                    if len(pids) >= n_threads:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Reached {n_threads} files, waiting for processing...")
                        for process in pids:
                            result = process.wait()
                            if result != 0:
                                result_all = 1
                                if verbosity > VerbosityLevel.BASIC:
                                    print(f"Error occurred while processing a file!")
                        pids = []
                        if verbosity > VerbosityLevel.BASIC:
                            print("Processed files, continuing...")

                # Wait for remaining processes
                for process in pids:
                    result = process.wait()
                    if result != 0:
                        result_all = 1
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error occurred while processing a file!")

                if result_all != 0:
                    raise RuntimeError("Errors occurred during event2image processing")

                if verbosity > VerbosityLevel.BASIC:
                    print(f"Finished processing {file_cnt} .empirevent files")
                    print(f"Total processing time: {time.time() - start_time:.2f} seconds")