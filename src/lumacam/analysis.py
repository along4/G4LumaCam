import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass
import json
from typing import Dict, Any, Optional, Union

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

    def write(self, output_file: str=".parameterSettings.json") -> str:
        """
        Write the photon2event configuration to a JSON file.
        
        Args:
            output_file: The path to save the parameters file.
            
        Returns:
            The path to the created JSON file.
        """
        parameters = {
            "photon2event": {
                "dSpace_px": self.dSpace_px,
                "dTime_s": self.dTime_s,
                "durationMax_s": self.durationMax_s,
                "dTime_ext": self.dTime_ext
            }
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
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
        self.binning_t_relToExtTrigger = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def psd_binning(self) -> 'EventBinningConfig':
        self.binning_psd = BinningConfig(
            resolution=1e-6,
            nBins=100,
            offset=0
        )
        return self
    
    def nphotons_binning(self) -> 'EventBinningConfig':
        self.binning_nPhotons = BinningConfig(
            resolution=1,
            nBins=10,
            offset=0
        )
        return self
    
    def time_binning(self) -> 'EventBinningConfig':
        self.binning_t = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def spatial_binning(self) -> 'EventBinningConfig':
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
        parameters = {"bin_events": {}}
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
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        return output_file
    
    def _get_config_dict(self, config: BinningConfig, use_s: bool = False, use_px: bool = False) -> Dict[str, Any]:
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
    def __init__(self, archive: str = "test", empir_dirpath: str = None):
        """
        Analysis class for processing TPX3 data through the EMPIR pipeline.

        Args:
            archive: The directory containing TPX3 data or a groupby subfolder.
            empir_dirpath: Path to the EMPIR directory. If None, defaults to "./empir".
        """
        self.archive = Path(archive)
        
        # Check if this is a groupby folder
        self._is_groupby = False
        self._groupby_metadata = None
        self._groupby_subfolders = []
        
        # Look for groupby metadata in the archive or parent
        metadata_file = self.archive / ".groupby_metadata.json"
        if metadata_file.exists():
            self._is_groupby = True
            with open(metadata_file, 'r') as f:
                self._groupby_metadata = json.load(f)
            
            # Find all subfolders with SimPhotons or tpx3Files
            for subfolder in sorted(self.archive.iterdir()):
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    if (subfolder / "SimPhotons").exists() or (subfolder / "tpx3Files").exists():
                        self._groupby_subfolders.append(subfolder)
            
            print(f"Detected groupby structure: {len(self._groupby_subfolders)} groups")
            print(f"Groupby column: {self._groupby_metadata.get('column', 'unknown')}")
        
        if empir_dirpath is not None:
            self.empir_dirpath = Path(empir_dirpath)
        else:
            try:
                from G4LumaCam.config.paths import EMPIR_PATH
                self.empir_dirpath = Path(EMPIR_PATH)
            except ImportError:
                self.empir_dirpath = Path("./empir")
        
        # DON'T set photon_files_dir here if it's a groupby structure
        # It will be set per-group in process_grouped
        if not self._is_groupby:
            self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
        else:
            # For groupby, we'll set this per group
            self.photon_files_dir = None

        self.Photon2EventConfig = Photon2EventConfig
        self.EventBinningConfig = EventBinningConfig

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

        self.events_df = None
        self.photons_df = None
        self.associated_df = None

        self.default_params = {
            "in_focus": {
                "pixel2photon": {
                    "dSpace": 2,
                    "dTime": 100e-09,
                    "nPxMin": 8,
                    "nPxMax": 100,
                    "TDC1": true
                },
                "photon2event": {
                    "dSpace_px": 0.001,
                    "dTime_s": 5e-08,
                    "durationMax_s": 5e-07,
                    "dTime_ext": 5
                },
                "event2image": {
                    "size_x": 512,
                    "size_y": 512,
                    "nPhotons_min": 1,
                    "nPhotons_max": 1,
                    "psd_min": 0,
                    "time_extTrigger": "reference",
                    "time_res_s": 1.5625e-09,
                    "time_limit": 640
                },
            },
                "pixel2photon": {
                    "dSpace": 2,
                    "dTime": 5e-08,
                    "nPxMin": 2,
                    "nPxMax": 12,
                    "TDC1": true
                },
                "photon2event": {
                    "dSpace_px": 60,
                    "dTime_s": 10e-08,
                    "durationMax_s": 10e-07,
                    "dTime_ext": 5
                },
                "event2image": {
                    "size_x": 512,
                    "size_y": 512,
                    "nPhotons_min": 2,
                    "nPhotons_max": 9999,
                    "psd_min": 0,
                    "time_extTrigger": "reference",
                    "time_res_s": 1.5625e-09,
                    "time_limit": 640
                },
            },
            "hitmap": {
                "pixel2photon": {
                    "dSpace": 0.001,
                    "dTime":1e-9,
                    "nPxMin": 1,
                    "TDC1": true
                },
                "photon2event": {
                    "dSpace_px": 0.001,
                    "dTime_s": 0,
                    "durationMax_s": 0,
                    "dTime_ext": 5
                },
                "event2image": {
                    "size_x": 256,
                    "size_y": 256,
                    "nPhotons_min": 1,
                    "nPhotons_max": 9999,
                    "psd_min": 0,
                    "time_extTrigger": "reference",
                    "time_res_s": 1.5625e-09,
                    "time_limit": 640
                },
            }
        }

    def _run_pixel2photon(self, tpx3_dir: Path, photon_files_dir: Path, 
                                params_file: Path, n_threads: int, 
                                parameters: Dict[str, Any], 
                                verbosity: VerbosityLevel) -> None:
        """Run pixel2photon processing on TPX3 files.
        Args:
            tpx3_dir: Directory containing .tpx3 files.
            photon_files_dir: Directory to save .empirphot files.
            params_file: Path to the parameter settings JSON file.
            n_threads: Number of threads to use for parallel processing.
            parameters: Dictionary of parameters for pixel2photon.
            verbosity: Controls the level of output during processing.
        """
        tpx3_files = sorted([f for f in tpx3_dir.glob("*.tpx3")])
        if not tpx3_files:
            raise FileNotFoundError(f"No .tpx3 files with '_part' found in {tpx3_dir}")

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

        for process in pids:
            result = process.wait()
            if result != 0:
                result_all = 1
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error occurred while processing a file!")

        if result_all != 0:
            # raise RuntimeError("Errors occurred during pixel2photon processing")
            if verbosity > VerbosityLevel.BASIC:
                print("Errors occurred during pixel2photon processing")

        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished processing {file_cnt} .tpx3 files")

    def _run_photon2event(self, photon_files_dir: Path, event_files_dir: Path, params_file: Path, n_threads: int, verbosity: VerbosityLevel) -> None:
        """Run photon2event processing on .empirphot files.
        
            args:
                photon_files_dir: Directory containing .empirphot files.
                event_files_dir: Directory to save .empirevent files.
                params_file: Path to the parameter settings JSON file.
                n_threads: Number of threads to use for parallel processing.
                verbosity: Controls the level of output during processing.
        """
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

    def _run_event2image(self, event_files_dir: Path, final_dir: Path, params_file: Path, n_threads: int, verbosity: VerbosityLevel) -> None:
        """Run event2image processing on all .empirevent files to produce a single image.
        
        Args:
            event_files_dir: Directory containing .empirevent files.
            final_dir: Directory to save the final .empirimage file.
            params_file: Path to the parameter settings JSON file.
            n_threads: Number of threads to use for parallel processing (not used in this implementation).
            verbosity: Controls the level of output during processing.
        """
        empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
        if not empirevent_files:
            raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")

        if verbosity > VerbosityLevel.BASIC:
            print(f"Processing {len(empirevent_files)} .empirevent files into a single image...")

        output_file = final_dir / "combined.empirimage"
        cmd = [
            str(self.empir_dirpath / "bin/empir_event2image"),
            "-I", str(event_files_dir),
            "-o", str(output_file),
            "--paramsFile", str(params_file),
            # "--fileFormat", "empirimage"
        ]
        
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Running: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True)
        else:
            process = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if process.returncode != 0:
            raise RuntimeError("Error occurred during event2image processing")

        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished producing combined image: {output_file}")

    def _run_export_photons(self, process_dir: Path, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirphot files from photonFiles subfolder to CSV files in ExportedPhotons subfolder.
        
        Args:
            process_dir: Path to the processing directory (supports suffix).
            verbosity: Controls the level of output during processing.
        """
        photon_files_dir = process_dir / "photonFiles"
        if not photon_files_dir.exists():
            raise FileNotFoundError(f"{photon_files_dir} does not exist.")
        
        exported_photons_dir = process_dir / "ExportedPhotons"
        exported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        empirphot_files = sorted(photon_files_dir.glob("*.empirphot"))
        if not empirphot_files:
            raise FileNotFoundError(f"No .empirphot files found in {photon_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirphot_files)} .empirphot files to CSV...")
        
        for empirphot_file in tqdm(empirphot_files, desc="Exporting photons", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                photon_result_csv = exported_photons_dir / f"exported_{empirphot_file.stem}.csv"
                cmd = [
                    str(self.empir_dirpath / "empir_export_photons"),
                    str(empirphot_file),
                    str(photon_result_csv),
                    "csv"
                ]
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                try:
                    import pandas as pd
                    df = pd.read_csv(photon_result_csv)
                    df.columns = ["x", "y", "toa", "tof"]
                    df["x"] = df["x"].astype(float)
                    df["y"] = df["y"].astype(float)
                    df["toa"] = df["toa"].astype(float)
                    df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
                    df.to_csv(photon_result_csv, index=False)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✔ Exported and modified {empirphot_file.name} → {photon_result_csv.name}")
                        
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Error modifying headers for {photon_result_csv.name}: {e}")
                    
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"❌ Error exporting {empirphot_file.name}: {e}")
        
        if verbosity > VerbosityLevel.BASIC:
            print("✅ Finished exporting and modifying all photon files!")

    def _run_export_events(self, process_dir: Path, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirevent files from eventFiles subfolder to CSV files in ExportedEvents subfolder.
        
        Args:
            process_dir: Path to the processing directory (supports suffix).
            verbosity: Controls the level of output during processing.
        """
        event_files_dir = process_dir / "eventFiles"
        if not event_files_dir.exists():
            raise FileNotFoundError(f"{event_files_dir} does not exist.")
        
        exported_events_dir = process_dir / "ExportedEvents"
        exported_events_dir.mkdir(parents=True, exist_ok=True)
        
        empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
        if not empirevent_files:
            raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirevent_files)} .empirevent files to CSV...")
        
        for empirevent_file in tqdm(empirevent_files, desc="Exporting events", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                event_result_csv = exported_events_dir / f"{empirevent_file.stem}.csv"
                cmd = [
                    str(self.empir_dirpath / "empir_export_events"),
                    str(empirevent_file),
                    str(event_result_csv),
                    "csv"
                ]
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                try:
                    import pandas as pd
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
        Process TPX3 files through the EMPIR pipeline.
        
        Automatically detects groupby structures and processes all groups if found.
        
        Args:
            params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder (ignored for groupby structures)
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            verbosity: Controls output level
            clean: If True, deletes existing processed files
            **kwargs: Additional parameters
        """
        # Auto-detect and handle groupby structure
        if self._is_groupby:
            if verbosity > VerbosityLevel.BASIC:
                print("Detected groupby structure, processing all groups...")
            return self._process_grouped(
                params=params,
                n_threads=n_threads,
                suffix=suffix,  # Pass the suffix to _process_grouped
                pixel2photon=pixel2photon,
                photon2event=photon2event,
                event2image=event2image,
                export_photons=export_photons,
                export_events=export_events,
                verbosity=verbosity,
                clean=clean,
                **kwargs
            )
        
        # Call the single processing method
        return self._process_single(
            params=params,
            n_threads=n_threads,
            suffix=suffix,
            pixel2photon=pixel2photon,
            photon2event=photon2event,
            event2image=event2image,
            export_photons=export_photons,
            export_events=export_events,
            verbosity=verbosity,
            clean=clean,
            **kwargs
        )

    def _process_grouped(self, 
                        params: Union[str, Dict[str, Any]] = None,
                        n_threads: int = 1,
                        suffix: str = "",  # Added suffix parameter
                        pixel2photon: bool = True,
                        photon2event: bool = True,
                        event2image: bool = False,
                        export_photons: bool = True,
                        export_events: bool = False,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                        clean: bool = True,
                        **kwargs) -> None:
        """
        Process TPX3 files for all groups in a groupby structure.
        
        Args:
            params: Parameters for processing (same as process())
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder within each group
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            verbosity: Controls output level
            clean: If True, deletes existing processed files before processing
            **kwargs: Additional parameters
        
        Raises:
            ValueError: If not a groupby structure
        """
        if not self._is_groupby:
            raise ValueError("This archive is not a groupby structure. Use process() instead.")
        
        if not self._groupby_subfolders:
            raise ValueError("No valid group subfolders found.")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"Processing {len(self._groupby_subfolders)} groups")
            print(f"Column: {self._groupby_metadata.get('column', 'unknown')}")
            print(f"{'='*60}\n")
        
        # Store original archive
        original_archive = self.archive
        original_photon_files_dir = self.photon_files_dir
        
        # Process each group - show progress bar at BASIC level or higher
        group_iter = enumerate(self._groupby_subfolders)
        if verbosity >= VerbosityLevel.BASIC:
            group_iter = enumerate(tqdm(self._groupby_subfolders, 
                                        desc=f"Processing groups",
                                        position=0,
                                        leave=True))
        
        for i, group_folder in group_iter:
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\n{'─'*60}")
                print(f"Group {i+1}/{len(self._groupby_subfolders)}: {group_folder.name}")
                print(f"{'─'*60}")
            
            # Temporarily change archive to this group
            self.archive = group_folder
            # Set photon_files_dir to the suffixed subfolder's photonFiles directory
            if suffix:
                self.photon_files_dir = self.archive / suffix.strip("_") / "photonFiles"
            else:
                self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Check if tpx3Files exists
                tpx3_dir = group_folder / "tpx3Files"
                if not tpx3_dir.exists() or not list(tpx3_dir.glob("*.tpx3")):
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  No TPX3 files found in {group_folder.name}, skipping")
                    continue
                
                # Call _process_single directly (NOT process) to avoid recursion
                # Pass the suffix to create the suffixed subfolder within the group
                self._process_single(
                    params=params,
                    n_threads=n_threads,
                    suffix=suffix,  # Pass the suffix to _process_single
                    pixel2photon=pixel2photon,
                    photon2event=photon2event,
                    event2image=event2image,
                    export_photons=export_photons,
                    export_events=export_events,
                    verbosity=VerbosityLevel.QUIET,  # Always QUIET for internal processing
                    clean=clean,
                    **kwargs
                )
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"✓ Completed group '{group_folder.name}'")
            
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"\n✗ Error processing group '{group_folder.name}': {e}")
                if verbosity > VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
            
            finally:
                # Restore original archive
                self.archive = original_archive
                self.photon_files_dir = original_photon_files_dir
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"✓ Completed all {len(self._groupby_subfolders)} groups")
            print(f"{'='*60}\n")

    def _process_single(self, 
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
        Process TPX3 files through the EMPIR pipeline.
        
        Automatically detects groupby structures and processes all groups if found.
        
        Args:
            params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder (ignored for groupby structures)
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            verbosity: Controls output level
            clean: If True, deletes existing processed files
            **kwargs: Additional parameters
        """
            
        import time
        start_time = time.time()

        base_dir = self.archive
        if suffix:
            process_dir = base_dir / suffix.strip("_")
            process_dir.mkdir(parents=True, exist_ok=True)
            tpx3_dir = process_dir / "tpx3Files"
            tpx3_dir.mkdir(parents=True, exist_ok=True)
            
            orig_tpx3_dir = base_dir / "tpx3Files"
            if not orig_tpx3_dir.exists():
                raise FileNotFoundError(f"Original tpx3Files directory not found at {orig_tpx3_dir}")
            
            # Get list of existing .tpx3 files in the original directory
            existing_tpx3_files = sorted(orig_tpx3_dir.glob("*.tpx3"))
            if not existing_tpx3_files:
                raise FileNotFoundError(f"No .tpx3 files found in {orig_tpx3_dir}")
            
            current_dir = os.getcwd()
            try:
                for tpx3_file in existing_tpx3_files:
                    dest_file = tpx3_dir / tpx3_file.name
                    if not dest_file.exists() and tpx3_file.is_file():
                        os.symlink(tpx3_file.absolute(), dest_file)
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Created symlink: {dest_file} -> {tpx3_file.absolute()}")
                    elif not tpx3_file.is_file():
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Skipped invalid file: {tpx3_file} (not a regular file)")
                    elif dest_file.exists():
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Symlink already exists: {dest_file}")
            except OSError as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error creating symlink for {tpx3_file}: {e}")
            finally:
                os.chdir(current_dir)
        else:
            process_dir = base_dir
            tpx3_dir = base_dir / "tpx3Files"

        photon_files_dir = process_dir / "photonFiles"
        event_files_dir = process_dir / "eventFiles"
        final_dir = process_dir / "final"
        photon_files_dir.mkdir(parents=True, exist_ok=True)
        event_files_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

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

        if params is None:
            parameters = self.default_params.get("in_focus", {})
        elif isinstance(params, str):
            if params in ["in_focus", "out_of_focus", "hitmap"]:
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
            raise ValueError("params must be 'in_focus', 'out_of_focus', 'hitmap', a JSON file path, JSON string, or dictionary")

        if kwargs:
            if "pixel2photon" in parameters:
                parameters["pixel2photon"].update({k: v for k, v in kwargs.items() if k in ["dSpace", "dTime", "nPxMin", "nPxMax", "TDC1"]})
            if "photon2event" in parameters:
                parameters["photon2event"].update({k: v for k, v in kwargs.items() if k in ["dSpace_px", "dTime_s", "durationMax_s", "dTime_ext"]})
            if "event2image" in parameters:
                parameters["event2image"].update({k: v for k, v in kwargs.items() if k in ["size_x", "size_y", "nPhotons_min", "nPhotons_max", "time_res_s", "time_limit", "psd_min", "time_extTrigger"]})

        params_file = process_dir / ".parameterSettings.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Parameters written to {params_file}")

        if pixel2photon:
            self._run_pixel2photon(tpx3_dir, photon_files_dir, params_file, n_threads, parameters, verbosity)
        
        if photon2event:
            self._run_photon2event(photon_files_dir, event_files_dir, params_file, n_threads, verbosity)
        
        if export_photons:
            self._run_export_photons(process_dir, verbosity=verbosity)
        
        if export_events:
            self._run_export_events(process_dir, verbosity=verbosity)
        
        if event2image:
            self._run_event2image(event_files_dir, final_dir, params_file, n_threads, verbosity)
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")