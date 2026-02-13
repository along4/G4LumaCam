import os
import subprocess
import re
from dataclasses import dataclass
from importlib import resources
import shutil
from typing import Optional, Tuple, List
from pathlib import Path
from enum import IntEnum
from tqdm.notebook import tqdm
import pandas as pd
import threading
import queue
import time
import glob
from .g4config import Config

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

class Simulate:
    """Class to simulate the lumacam executable."""
    def __init__(self, archive: str = "archive/test"):
        """
        Initialize the Simulate object.
        """
        self.archive = Path(archive).absolute()
        self.archive.mkdir(exist_ok=True, parents=True)
        
        self.sim_dir = self.archive / "SimPhotons"
        self.sim_dir.mkdir(exist_ok=True, parents=True)

        self.lumacam_executable = self._find_lumacam_executable()

    def _find_lumacam_executable(self) -> str:
        """Resolve the lumacam executable path with sensible fallbacks.

        Priority:
        1) LUMACAM_EXECUTABLE env var (explicit override)
        2) Local build output: <repo>/build/lumacam
        3) Installed package resource: G4LumaCam/bin/lumacam
        """
        env_path = os.environ.get("LUMACAM_EXECUTABLE")
        if env_path and os.path.exists(env_path):
            return env_path

        repo_root = Path(__file__).resolve().parents[2]
        build_candidate = repo_root / "build" / "lumacam"
        if build_candidate.exists():
            return str(build_candidate)

        try:
            pkg_bin = resources.files("G4LumaCam") / "bin" / "lumacam"
            if pkg_bin.exists():
                return str(pkg_bin)
        except Exception:
            pass

        raise FileNotFoundError(
            f"lumacam executable not found. Tried env LUMACAM_EXECUTABLE, {build_candidate}, and package bin."
        )

    def _process_output(self, process, output_queue, verbosity):
        """Process the output from the simulation in real-time."""
        event_pattern = re.compile(r'--> Event (\d+) starts\.')
        final_event_pattern = re.compile(r'Simulating Event: (\d+)')
        cleanup_pattern = re.compile(r'Graphics systems deleted\.')
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            line = line.strip()
            if line:
                match = event_pattern.match(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                match = final_event_pattern.search(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                if cleanup_pattern.search(line):
                    output_queue.put(('complete', None))
                
                if verbosity >= VerbosityLevel.DETAILED:
                    output_queue.put(('output', line))
                elif verbosity >= VerbosityLevel.BASIC and ('starts.' in line or 'Run' in line or 'G4Exception' in line):
                    output_queue.put(('output', line))

    def clear_subfolders(self, verbosity: VerbosityLevel = VerbosityLevel.BASIC):
        """Remove all contents of SimPhotons subfolder if it exists.
        This ensures that old simulation data does not interfere with new runs.
        Args:
            verbosity (VerbosityLevel): Level of verbosity for print statements.           
        """
        if self.sim_dir.exists():
            for item in self.sim_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Cleared contents of {self.sim_dir}")

    def run(self, 
            config_or_file: Optional[str | Config] = None, 
            verbosity: VerbosityLevel = VerbosityLevel.BASIC) -> pd.DataFrame:
        """
        Run the lumacam executable with either a Config object or a macro file.
        """
        if not os.path.exists(self.lumacam_executable):
            raise FileNotFoundError(f"lumacam executable not found at {self.lumacam_executable}")

        # Check write permissions
        if not os.access(self.sim_dir, os.W_OK):
            raise PermissionError(f"No write permission in {self.sim_dir}")

        # Clear SimPhotons subfolder
        self.clear_subfolders(verbosity=verbosity)

        temp_macro = None
        macro_file = None
        num_events = None
        progress_interval = None
        csv_filename = "sim_data.csv"
        
        if isinstance(config_or_file, Config):
            temp_macro = self.sim_dir / "macro.mac"
            macro_file = config_or_file.write(str(temp_macro))
            num_events = config_or_file.num_events
            progress_interval = config_or_file.progress_interval
            csv_filename = config_or_file.csv_filename
            shutil.copy(str(temp_macro), str(self.archive / "macro.mac"))
        elif isinstance(config_or_file, str):
            if not os.path.exists(config_or_file):
                raise FileNotFoundError(f"Macro file not found at {config_or_file}")
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
                match = re.search(r'/lumacam/csvFilename\s+(\S+)', content)
                if match:
                    csv_filename = match.group(1)

        original_dir = os.getcwd()
        os.chdir(str(self.archive))

        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Expected CSV output: {csv_filename}")

        try:
            process = subprocess.Popen(
                [self.lumacam_executable, "macro.mac"],
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
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(content)
                except queue.Empty:
                    continue

            if pbar is not None:
                if run_completed or last_event >= num_events - 1:
                    pbar.n = num_events
                pbar.refresh()
                pbar.close()

            stderr = process.stderr.read()
            if process.returncode != 0 and verbosity > 1:
                print(f"lumacam execution failed with error:\n{stderr}")

            base_name = csv_filename.rsplit('.', 1)[0]
            extension = csv_filename.rsplit('.', 1)[1] if '.' in csv_filename else "csv"
            
            dfs = []
            # Use glob to find all CSV files in SimPhotons directory
            csv_pattern = os.path.join(str(self.sim_dir), f"{base_name}*.{extension}")
            csv_files = sorted(glob.glob(csv_pattern))
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Looking for CSV files with pattern: {csv_pattern}")
                print(f"Found CSV files: {csv_files}")
            
            for csv_file in csv_files:
                csv_path = Path(csv_file)
                if csv_path.exists():
                    file_size = csv_path.stat().st_size
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Processing CSV file: {csv_path} (size: {file_size} bytes)")
                    
                    if file_size == 0:
                        csv_path.unlink()
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Removed empty CSV file: {csv_path}")
                        continue
                    
                    try:
                        df = pd.read_csv(csv_path)
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"CSV file {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        if df.shape[0] == 0:
                            csv_path.unlink()
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Removed header-only CSV file: {csv_path}")
                        else:
                            dfs.append(df)
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Added {df.shape[0]} rows from {csv_path}")
                    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Removing malformed/empty CSV file {csv_path}: {e}")
                        csv_path.unlink()
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Error reading CSV file {csv_path}: {e}")
                        csv_path.unlink()
                else:
                    print(f"CSV file does not exist: {csv_path}")
            
            if not dfs:
                print(f"No valid (non-empty) CSV files found in {self.sim_dir}. Check EventProcessor output logic or simulation configuration.")
                return pd.DataFrame()  # Return empty DataFrame instead of raising an error
            
            combined_df = pd.concat(dfs, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Combined DataFrame: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            return combined_df

        finally:
            os.chdir(original_dir)
