from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec, EvenPolynomial, InteractiveLayout
from rayoptics.gui import roafile
from rayoptics.elem.elements import Element
from rayoptics.raytr.trace import apply_paraxial_vignetting
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path
from multiprocessing import Pool
from functools import partial   
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from rayoptics.raytr import analyses
from lmfit import Parameters, minimize, MinimizerException
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import glob

from enum import IntEnum

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything


class Lens:
    """
    Lens defining object with integrated data management
    """
    def __init__(self, archive: str = None, data: "pd.DataFrame" = None,
                 kind="Nikkor_58mm_095", dist_from_screen: float = 461.535):
        """
        Lens constructor
        Input:
            archive: str, Optional, Name of the archive directory for saving results
            data: pd.DataFrame, Optional, optical photon data table
            kind: str, Optional, Lens kind, supported lenses are ['Nikkor_58mm_095']
            dist_from_screen: float, Optional, Distance from the screen in mm
        """
        self.dist_from_screen = dist_from_screen
        self.kind = kind

        if archive is not None:
            self.archive = Path(archive)
            self.archive.mkdir(parents=True, exist_ok=True)

            # Load all sim_data_?.csv files from SimPhotons directory
            sim_photons_dir = self.archive / "SimPhotons"
            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))

            valid_dfs = []
            # print(f"Found {len(csv_files)} simulation data files. Loading...")

            for file in tqdm(csv_files, desc="Loading simulation data"):
                try:
                    if file.stat().st_size > 100:  # Ignore empty files
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    # print(f"⚠️ Skipping {file.name} due to error: {e}")
                    pass

            if valid_dfs:
                self.data = pd.concat(valid_dfs, ignore_index=True)
            else:
                print("No valid simulation data files found, initializing empty DataFrame.")
                self.data = pd.DataFrame()

        elif data is not None:
            self.data = data
            self.archive = Path("archive/test")
        else:
            raise ValueError("Either archive or data must be provided")

        # Initialize optical model
        self.opm = None
        if self.kind == "Nikkor_58mm_095":
            self.opm0 = self.nikkor_58mm(save=False)
        else:
            raise ValueError(f"Unknown lens kind: {kind}, supported lenses are ['Nikkor_58mm_095']")


    def nikkor_58mm(self, save: bool = False):
        """
        Nikkor 58mm f/0.95 lens
        
        Input:
        save: bool
            Save the optical model to a file
        Returns:
            OpticalModel
        """

        # WO2019-229849 Example 1 (Nikkor Z 58mm f/0.95 S)
        # Obtained via https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm

        # Surface[1] = type=S radius=108.488 thickness=7.65 diameter = 66.8 nd = 1.90265 vd = 35.77
        # Aspheric values[1] = 108.488 0 -3.82177e-07 -6.06486e-11 -3.80172e-15 -1.32266e-18
        # Surface[2] = type=S radius=-848.55 thickness=2.8 diameter = 65.82 nd = 1.552981 vd = 55.07
        # Surface[3] = type=S radius=50.252 thickness=18.12 diameter = 57.94 nd = 0 vd = 0
        # Surface[4] = type=S radius=-60.72 thickness=2.8 diameter = 58.28 nd = 1.61266 vd = 44.46
        # Surface[5] = type=S radius=2497.5 thickness=9.15 diameter = 65.32 nd = 1.59319 vd = 67.9
        # Surface[6] = type=S radius=-77.239 thickness=0.4 diameter = 65.32 nd = 0 vd = 0
        # Surface[7] = type=S radius=113.763 thickness=10.95 diameter = 70.9 nd = 1.8485 vd = 43.79
        # Surface[8] = type=S radius=-178.06 thickness=0.4 diameter = 70.9 nd = 0 vd = 0
        # Surface[9] = type=S radius=70.659 thickness=9.74 diameter = 65 nd = 1.59319 vd = 67.9
        # Surface[10] = type=S radius=-1968.5 thickness=0.2 diameter = 65 nd = 0 vd = 0
        # Surface[11] = type=S radius=289.687 thickness=8 diameter = 61.06 nd = 1.59319 vd = 67.9
        # Surface[12] = type=S radius=-97.087 thickness=2.8 diameter = 59.42 nd = 1.738 vd = 32.33
        # Surface[13] = type=S radius=47.074 thickness=8.7 diameter = 50.24 nd = 0 vd = 0
        # Surface[14] = type=AS radius=0 thickness=5.29 diameter = 47.918 nd = 0 vd = 0
        # Surface[15] = type=S radius=-95.23 thickness=2.2 diameter = 49.92 nd = 1.61266 vd = 44.46
        # Surface[16] = type=S radius=41.204 thickness=11.55 diameter = 49.92 nd = 1.49782 vd = 82.57
        # Surface[17] = type=S radius=-273.092 thickness=0.2 diameter = 49.92 nd = 0 vd = 0
        # Surface[18] = type=S radius=76.173 thickness=9.5 diameter = 51.12 nd = 1.883 vd = 40.69
        # Surface[19] = type=S radius=-101.575 thickness=0.2 diameter = 51.12 nd = 0 vd = 0
        # Surface[20] = type=S radius=176.128 thickness=7.45 diameter = 46.8 nd = 1.95375 vd = 32.33
        # Aspheric values[20] = 176.128 0 -1.15028e-06 -4.51771e-10 2.7267e-13 -7.66812e-17
        # Surface[21] = type=S radius=-67.221 thickness=1.8 diameter = 45.36 nd = 1.738 vd = 32.33
        # Surface[22] = type=S radius=55.51 thickness=2.68 diameter = 39.84 nd = 0 vd = 0
        # Surface[23] = type=S radius=71.413 thickness=6.35 diameter = 39.46 nd = 1.883 vd = 40.69
        # Surface[24] = type=S radius=-115.025 thickness=1.81 diameter = 39.46 nd = 1.69895 vd = 30.13
        # Surface[25] = type=S radius=46.943 thickness=0.8 diameter = 39.46 nd = 0 vd = 0
        # Surface[26] = type=S radius=55.281 thickness=9.11 diameter = 38.94 nd = 1.883 vd = 40.69
        # Surface[27] = type=S radius=-144.041 thickness=3 diameter = 38.28 nd = 1.765538 vd = 46.76
        # Surface[28] = type=S radius=52.858 thickness=14.5 diameter = 38.28 nd = 0 vd = 0
        # Aspheric values[28] = 52.858 0 3.18645e-06 -1.14718e-08 7.74567e-11 -2.24225e-13 3.3479e-16 -1.7047e-19
        # Surface[29] = type=S radius=0 thickness=1.6 diameter = 44.3 nd = 1.5168 vd = 64.14
        # Surface[30] = type=S radius=0 thickness=1 diameter = 44.3 nd = 0 vd = 0

        opm = OpticalModel()
        sm  = opm.seq_model
        osp = opm.optical_spec
        pm = opm.parax_model
        osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=0.98)
        osp.field_of_view = FieldSpec(osp, key=['object', 'angle'], flds=[0., 19.98])
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        # osp.spectral_region = WvlSpec([(300, 0.1),(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        opm.system_spec.title = 'WO2019-229849 Example 1 (Nikkor Z 58mm f/0.95 S)'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True
        sm.gaps[0].thi = self.dist_from_screen # mm
        sm.add_surface([108.488,7.65,1.90265,35.77])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=108.488, cc=0,
                coefs=[0.0,-3.82177e-07,-6.06486e-11,-3.80172e-15,-1.32266e-18,0,0])
        sm.ifcs[sm.cur_surface].max_aperture = 33.4
        sm.add_surface([-848.55,2.8,1.55298,55.07])
        sm.ifcs[sm.cur_surface].max_aperture = 32.91
        sm.add_surface([50.252,18.12])
        sm.ifcs[sm.cur_surface].max_aperture = 28.97
        sm.add_surface([-60.72,2.8,1.61266,44.46])
        sm.ifcs[sm.cur_surface].max_aperture = 29.14
        sm.add_surface([2497.5,9.15,1.59319,67.9])
        sm.ifcs[sm.cur_surface].max_aperture = 32.66
        sm.add_surface([-77.239,0.4])
        sm.ifcs[sm.cur_surface].max_aperture = 32.66
        sm.add_surface([113.763,10.95,1.8485,43.79])
        sm.ifcs[sm.cur_surface].max_aperture = 35.45
        sm.add_surface([-178.06,0.4])
        sm.ifcs[sm.cur_surface].max_aperture = 35.45
        sm.add_surface([70.659,9.74,1.59319,67.9])
        sm.ifcs[sm.cur_surface].max_aperture = 32.5
        sm.add_surface([-1968.5,0.2])
        sm.ifcs[sm.cur_surface].max_aperture = 32.5
        sm.add_surface([289.687,8,1.59319,67.9])
        sm.ifcs[sm.cur_surface].max_aperture = 30.53
        sm.add_surface([-97.087,2.8,1.738,32.33])
        sm.ifcs[sm.cur_surface].max_aperture = 29.71
        sm.add_surface([47.074,8.7])
        sm.ifcs[sm.cur_surface].max_aperture = 25.12
        sm.add_surface([0,5.29])
        sm.set_stop()
        sm.ifcs[sm.cur_surface].max_aperture = 23.959
        sm.add_surface([-95.23,2.2,1.61266,44.46])
        sm.ifcs[sm.cur_surface].max_aperture = 24.96
        sm.add_surface([41.204,11.55,1.49782,82.57])
        sm.ifcs[sm.cur_surface].max_aperture = 24.96
        sm.add_surface([-273.092,0.2])
        sm.ifcs[sm.cur_surface].max_aperture = 24.96
        sm.add_surface([76.173,9.5,1.883,40.69])
        sm.ifcs[sm.cur_surface].max_aperture = 25.56
        sm.add_surface([-101.575,0.2])
        sm.ifcs[sm.cur_surface].max_aperture = 25.56
        sm.add_surface([176.128,7.45,1.95375,32.33])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=176.128, cc=0,
                coefs=[0.0,-1.15028e-06,-4.51771e-10,2.7267e-13,-7.66812e-17,0,0])
        sm.ifcs[sm.cur_surface].max_aperture = 23.4
        sm.add_surface([-67.221,1.8,1.738,32.33])
        sm.ifcs[sm.cur_surface].max_aperture = 22.68
        sm.add_surface([55.51,2.68])
        sm.ifcs[sm.cur_surface].max_aperture = 19.92
        sm.add_surface([71.413,6.35,1.883,40.69])
        sm.ifcs[sm.cur_surface].max_aperture = 19.73
        sm.add_surface([-115.025,1.81,1.69895,30.13])
        sm.ifcs[sm.cur_surface].max_aperture = 19.73
        sm.add_surface([46.943,0.8])
        sm.ifcs[sm.cur_surface].max_aperture = 19.73
        sm.add_surface([55.281,9.11,1.883,40.69])
        sm.ifcs[sm.cur_surface].max_aperture = 19.47
        sm.add_surface([-144.041,3,1.76554,46.76])
        sm.ifcs[sm.cur_surface].max_aperture = 19.14
        sm.add_surface([52.858,14.5])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=52.858, cc=0,
                coefs=[0.0,3.18645e-06,-1.14718e-08,7.74567e-11,-2.24225e-13,3.3479e-16,-1.7047e-19])
        sm.ifcs[sm.cur_surface].max_aperture = 19.14
        sm.add_surface([0,1.6,1.5168,64.14])
        sm.ifcs[sm.cur_surface].max_aperture = 22.15
        sm.add_surface([0,1])
        sm.ifcs[sm.cur_surface].max_aperture = 22.15
        # sm.list_surfaces()
        # sm.list_gaps()
        sm.do_apertures = False
        opm.update_model()
        apply_paraxial_vignetting(opm)
        if save:
            opm.save_model(self.archive / "Nikkor_58mm_095")
        return opm

    def refocus(self, opm: "OpticalModel"=None, zfocus=0, zfine=0, fnumber=None, save: bool=False):
        """
        Refocus the lens
        
        Input:
            - opm: OpticalModel, Optional, Optical model to refocus
            - zfocus: (float, Optional), Focus distance in mm
            - zfine: (float, Optional), Fine focus distance in mm
            - fnumber: (float, Optional), New f-number for the lens (None = no change)
            - save: bool, Optional, Save the optical model to a file
        
        Returns:
            - OpticalModel
        """
        from copy import deepcopy
        if opm == None:
            opm = self.opm0
        opm = deepcopy(opm)
        sm = opm.seq_model
        osp = opm.optical_spec
        
        Δ0 = sm.gaps[-9].thi
        Δ = zfine
        Δs = sm.gaps[-3].thi
        s = {}
        dfs = {}
        δ = zfocus
        sm.gaps[0].thi = self.dist_from_screen + δ - Δ
        sm.gaps[-9].thi = Δ0 + Δ
        
        # Change the f-number if specified
        if fnumber is not None:
            # Set the pupil specification directly
            osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber)
        
        # Update the model with the new settings
        sm.do_apertures = False
        opm.update_model()
        apply_paraxial_vignetting(opm)
        
        if save:
            fnumber_str = f"_f{fnumber:.2f}" if fnumber is not None else ""
            save_path = self.archive / f"refocus_zfocus_{zfocus}_zfine_{zfine}{fnumber_str}"
            opm.save_model(save_path)
        return opm

    def _chunk_rays(self, rays, chunk_size=1000):
        """Split rays into chunks to reduce multiprocessing overhead"""
        # Pre-calculate length to avoid repeated len() calls
        n_rays = len(rays)
        return [rays[i:i + chunk_size] for i in range(0, n_rays, chunk_size)]

    def _process_ray_chunk(self, chunk, opt_model):
        """Process a chunk of rays at once"""
        try:
            return analyses.trace_list_of_rays(
                opt_model,
                chunk,
                output_filter="last",
                rayerr_filter="summary"
            )
        except Exception as e:
            # Log the error and return empty results
            # print(f"Error processing chunk: {str(e)}")
            return [None] * len(chunk)




    def trace_rays(self, opm=None, join=False, print_stats=False, n_processes=None,
                chunk_size=1000, progress_bar=True, timeout=3600, return_df=False,
                verbosity=VerbosityLevel.BASIC):
        """
        Trace rays from simulation data files and save processed results.

        This method:
        1. Locates all non-empty 'sim_data_*.csv' files in 'SimPhotons' directory under self.archive
        2. Processes ray data in parallel chunks using the specified optical model
        3. Saves traced results to 'TracedPhotons' directory
        4. Optionally returns combined results as a DataFrame

        Parameters:
        -----------
        opm : OpticalModel, optional
            Custom optical model to use instead of self.opm0
        join : bool, default False
            If True, concatenates original data with traced results
        print_stats : bool, default False
            If True, prints tracing statistics
        n_processes : int, optional
            Number of processes for parallel execution (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        progress_bar : bool, default True
            If True, displays a progress bar during processing
        timeout : int, default 3600
            Maximum time in seconds for processing each file
        return_df : bool, default False
            If True, returns a combined DataFrame of all processed files
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls the level of output detail:
            - QUIET: Only progress bar
            - BASIC: Progress bar + basic info
            - DETAILED: All available information

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True, otherwise None

        Raises:
        -------
        Exception
            If parallel processing or file operations fail
        """
        sim_photons_dir = self.archive / "SimPhotons"
        traced_photons_dir = self.archive / "TracedPhotons"
        traced_photons_dir.mkdir(parents=True, exist_ok=True)

        # Find all non-empty sim_data_*.csv files
        csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
        valid_files = [f for f in csv_files if f.stat().st_size > 100]  # Adjust threshold as needed

        if not valid_files:
            if verbosity >= VerbosityLevel.BASIC:
                print("No valid simulation data files found in 'SimPhotons' directory.")
            return None

        all_results = []  # Store results for combining if return_df=True

        # Progress bar for file processing
        file_iter = tqdm(valid_files, desc="Processing files", disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
        
        for csv_file in file_iter:
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Processing file: {csv_file.name}")

            # Load data
            df = pd.read_csv(csv_file)
            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping empty file: {csv_file.name}")
                continue

            # Set up optical model
            opm = deepcopy(opm) if opm else deepcopy(self.opm0)
            wvl = df["wavelength"].value_counts().to_frame().reset_index()
            wvl["count"] = 1
            opm.optical_spec.spectral_region = WvlSpec(wvl.values, ref_wl=1)

            # Convert DataFrame to ray format
            rays = [
                (np.array([row.x, row.y, row.z], dtype=np.float64),
                np.array([row.dx, row.dy, row.dz], dtype=np.float64),
                np.array([row.wavelength], dtype=np.float64))
                for row in df.itertuples()
            ]

            # Split rays into chunks
            ray_chunks = self._chunk_rays(rays, chunk_size)
            rays = None  # Clear memory

            # Process chunks in parallel
            process_chunk = partial(self._process_ray_chunk, opt_model=opm)
            try:
                with Pool(processes=n_processes) as pool:
                    chunk_results = list(tqdm(pool.imap_unordered(process_chunk, ray_chunks),
                                            total=len(ray_chunks), desc=f"Tracing rays ({csv_file.name})",
                                            disable=not progress_bar or verbosity == VerbosityLevel.QUIET))
                    pool.close()
                    pool.join()
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error in parallel processing for {csv_file.name}: {str(e)}")
                raise

            # Process results
            processed_results = []
            for chunk_result in chunk_results:
                if chunk_result is None:
                    processed_results.append({"x2": np.nan, "y2": np.nan, "z2": np.nan})
                    continue
                for entry in chunk_result:
                    if entry is None:
                        processed_results.append({"x2": np.nan, "y2": np.nan, "z2": np.nan})
                    else:
                        try:
                            ray, path_length, wvl = entry
                            position = ray[0]
                            processed_results.append({
                                "x2": position[0], "y2": position[1], "z2": position[2]
                            })
                        except Exception as e:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Error processing result entry: {str(e)}")
                            processed_results.append({"x2": np.nan, "y2": np.nan, "z2": np.nan})

            # Create result DataFrame
            result = pd.DataFrame(processed_results)
            if "toa" in df.columns:
                result["toa2"] = df["toa"]  # Preserve original time of arrival

            if join:
                result = pd.concat([df, result], axis=1)

            if print_stats and verbosity >= VerbosityLevel.BASIC:
                total = len(df)
                traced = result.dropna(subset=["x2"]).shape[0]
                percentage = (traced / total) * 100
                print(f"File: {csv_file.name} - Original events: {total}, "
                    f"Traced events: {traced}, Percentage: {percentage:.1f}%")

            # Save results
            output_file = traced_photons_dir / f"traced_{csv_file.name}"
            result.to_csv(output_file, index=False)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Saved traced results to {output_file}")

            if return_df:
                all_results.append(result)

        # Return combined results if requested
        if return_df and all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Returning combined DataFrame with {len(combined_df)} rows")
            return combined_df

        return None


    def zscan(self, zfocus_range: Union[np.ndarray, list, float] = 0.,
            zfine_range: Union[np.ndarray, list, float] = 0.,
            data: pd.DataFrame = None, opm: "OpticalModel" = None,
            n_processes: int = None, chunk_size: int = 1000,
            archive: str = None, verbose: VerbosityLevel = VerbosityLevel.QUIET) -> pd.Series:
        """
        Perform a Z-scan to determine the optimal focus by evaluating ray tracing results.

        Parameters:
        -----------
        zfocus_range : Union[np.ndarray, list, float], default 0.
            Range of z-focus positions to scan (scalar or iterable)
        zfine_range : Union[np.ndarray, list, float], default 0.
            Range of z-fine positions to scan (scalar or iterable)
        data : pd.DataFrame, optional
            Input DataFrame with ray data; overrides archive/class data if provided
        opm : OpticalModel, optional
            Custom optical model; uses self.opm0 if None
        n_processes : int, optional
            Number of processes for parallel ray tracing (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        archive : str, optional
            Path to archive directory containing 'SimPhotons' with simulation data files
        verbose : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail: QUIET (0), BASIC (1), DETAILED (2)

        Returns:
        --------
        pd.Series
            Series mapping each scanned z-value to the combined standard deviation of x2 and y2
        """
        # Load data from archive if provided
        if archive is not None:
            archive_path = Path(archive)
            sim_photons_dir = archive_path / "SimPhotons"
            if not sim_photons_dir.exists():
                raise ValueError(f"SimPhotons directory not found in {archive_path}")

            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data", disable=verbose == VerbosityLevel.QUIET):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
                    continue

            if valid_dfs:
                data = pd.concat(valid_dfs, ignore_index=True)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Loaded {len(valid_dfs)} valid simulation data files with {len(data)} rows.")
            else:
                raise ValueError(f"No valid simulation data files found in {sim_photons_dir}")

        # Fallback to provided or class data
        if data is None:
            data = getattr(self, 'data', None)
            if data is None:
                raise ValueError("No data provided and no class data available.")
            if verbose >= VerbosityLevel.DETAILED:
                print(f"Using class data with {len(data)} rows.")

        opm = deepcopy(opm) if opm else deepcopy(self.opm0)

        # Determine scan type
        zfocus_is_scalar = isinstance(zfocus_range, (float, int))
        zfine_is_scalar = isinstance(zfine_range, (float, int))

        if not zfocus_is_scalar and not zfine_is_scalar:
            raise ValueError("Either zfocus_range or zfine_range must be a scalar, not both iterables.")

        if zfocus_is_scalar:
            scan_range = np.array(zfine_range if isinstance(zfine_range, (np.ndarray, list)) else [zfine_range])
            fixed_value = zfocus_range
            scan_type = "zfine"
        else:
            scan_range = np.array(zfocus_range if isinstance(zfocus_range, (np.ndarray, list)) else [zfocus_range])
            fixed_value = zfine_range
            scan_type = "zfocus"

        results = {}
        min_std = float('inf')
        best_focus = None

        # Single progress bar for zscan
        pbar = tqdm(scan_range, desc=f"Z-scan ({scan_type})", disable=verbose == VerbosityLevel.QUIET)

        for value in pbar:
            if verbose >= VerbosityLevel.DETAILED:
                print(f"\n{'='*50}")
                print(f"Processing {scan_type} = {value:.2f}")

            try:
                new_opm = self.refocus(opm, zfocus=value if scan_type == "zfocus" else fixed_value,
                                    zfine=value if scan_type == "zfine" else fixed_value)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Refocused OPM with {scan_type}={value:.2f}, fixed_value={fixed_value:.2f}")
            except Exception as e:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Error refocusing for {scan_type} = {value:.2f}: {str(e)}")
                results[value] = float('inf')
                continue

            # Trace rays with progress bar off for BASIC verbosity
            traced_df = self.trace_rays(opm=new_opm, join=False, print_stats=False,
                                    n_processes=n_processes, chunk_size=chunk_size,
                                    progress_bar=False,  # Disable nested bars
                                    return_df=True, verbosity=verbose)

            if traced_df is None or traced_df.empty:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Warning: No valid traced data for {scan_type} = {value:.2f}")
                results[value] = float('inf')
                continue

            if 'x2' not in traced_df.columns or 'y2' not in traced_df.columns:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Warning: Missing x2 or y2 columns for {scan_type} = {value:.2f}")
                results[value] = float('inf')
                continue

            try:
                valid_x2 = traced_df['x2'].dropna()
                valid_y2 = traced_df['y2'].dropna()
                if len(valid_x2) == 0 or len(valid_y2) == 0:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"No valid data points for std calculation at {scan_type} = {value:.2f}")
                    results[value] = float('inf')
                    continue

                std_value = np.sqrt(valid_x2.std()**2 + valid_y2.std()**2)
                results[value] = std_value

                if verbose >= VerbosityLevel.DETAILED:
                    print(f"x2 std: {valid_x2.std():.3f}, y2 std: {valid_y2.std():.3f}, combined std: {std_value:.3f}")

                if std_value < min_std:
                    min_std = std_value
                    best_focus = value

                if verbose >= VerbosityLevel.BASIC:
                    pbar.set_description(
                        f"Z-scan ({scan_type}) [current: {value:.2f}, std: {std_value:.3f}, best: {min_std:.3f} @ {best_focus:.2f}]"
                    )

            except Exception as e:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Error analyzing traced data for {scan_type} = {value:.2f}: {str(e)}")
                if verbose >= VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
                results[value] = float('inf')
                continue

        pbar.close()

        if verbose >= VerbosityLevel.DETAILED:
            print("\nFinal results:")
            for k, v in sorted(results.items()):
                print(f"{scan_type} = {k:.2f}: std = {v:.3f}")

        if min_std != float('inf') and best_focus is not None:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Z-scan completed. Best {scan_type}: {best_focus:.2f} with std: {min_std:.3f}")
        else:
            if verbose >= VerbosityLevel.BASIC:
                print("Z-scan completed but no valid results were found.")

        return pd.Series(results)




    def zscan_optimize(self, initial_zfocus: float = 0., initial_zfine: float = 0.,
                    initial_fnumber: float = None,
                    optimize_param: str = "zfocus", zfocus_min: float = None, zfocus_max: float = None,
                    zfine_min: float = None, zfine_max: float = None,
                    fnumber_min: float = None, fnumber_max: float = None,
                    data: pd.DataFrame = None, opm: "OpticalModel" = None,
                    n_processes: int = None, chunk_size: int = 1000, archive: str = None,
                    verbose: VerbosityLevel = VerbosityLevel.BASIC) -> dict:
        """
        Optimize z-focus, z-fine positions, and/or f-number using lmfit minimization to minimize ray position spread
        while maximizing the number of traced photons.

        This method:
        1. Loads simulation data from archive, provided DataFrame, or class data
        2. Optimizes zfocus, zfine, fnumber, or combinations sequentially using lmfit
        3. Balances minimizing std of x2/y2 with maximizing traced photons
        4. Returns the best parameters and minimum objective value achieved

        Parameters:
        -----------
        initial_zfocus : float, default 0.
            Initial guess for z-focus position
        initial_zfine : float, default 0.
            Initial guess for z-fine position
        initial_fnumber : float, optional
            Initial guess for f-number (None uses the default lens f-number)
        optimize_param : str, default "zfocus"
            Parameter to optimize: "zfocus", "zfine", "fnumber", or combinations like "both", 
            "zfocus+fnumber", "zfine+fnumber", "all"
        zfocus_min : float, optional
            Minimum allowable z-focus value
        zfocus_max : float, optional
            Maximum allowable z-focus value
        zfine_min : float, optional
            Minimum allowable z-fine value
        zfine_max : float, optional
            Maximum allowable z-fine value
        fnumber_min : float, optional
            Minimum allowable f-number value
        fnumber_max : float, optional
            Maximum allowable f-number value
        data : pd.DataFrame, optional
            Input ray data; overrides archive/class data if provided
        opm : OpticalModel, optional
            Custom optical model; uses self.opm0 if None
        n_processes : int, optional
            Number of processes for parallel ray tracing (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        archive : str, optional
            Path to archive directory with 'SimPhotons' simulation data files
        verbose : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail: QUIET (0), BASIC (1), DETAILED (2)

        Returns:
        --------
        dict
            Optimization results with keys:
            - best_zfocus: Optimal z-focus position (if optimized)
            - best_zfine: Optimal z-fine position (if optimized)
            - best_fnumber: Optimal f-number (if optimized)
            - min_std: Minimum standard deviation achieved
            - traced_fraction: Fraction of photons traced at optimal position
            - result: lmfit MinimizeResult object from the last optimization

        Raises:
        -------
        ValueError
            If optimize_param is invalid or no valid data is found
        """
        # Load data from archive if provided
        if archive is not None:
            archive_path = Path(archive)
            sim_photons_dir = archive_path / "SimPhotons"
            if not sim_photons_dir.exists():
                raise ValueError(f"SimPhotons directory not found in {archive_path}")

            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data", disable=verbose == VerbosityLevel.QUIET):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
                    continue

            if valid_dfs:
                data = pd.concat(valid_dfs, ignore_index=True)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Loaded {len(valid_dfs)} valid simulation data files with {len(data)} rows.")
            else:
                raise ValueError(f"No valid simulation data files found in {sim_photons_dir}")

        # Fallback to provided or class data
        if data is None:
            data = getattr(self, 'data', None)
            if data is None:
                raise ValueError("No data provided and no class data available.")
            if verbose >= VerbosityLevel.DETAILED:
                print(f"Using class data with {len(data)} rows.")

        # Clean data by dropping rows with NaN in essential columns
        essential_columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'wavelength']
        if not all(col in data.columns for col in essential_columns):
            raise ValueError(f"Data missing required columns: {essential_columns}")
        data = data.dropna(subset=essential_columns)
        if data.empty:
            raise ValueError("Data is empty after removing NaN from essential columns.")
        total_photons = len(data)
        if verbose >= VerbosityLevel.DETAILED:
            print(f"Cleaned data to {total_photons} rows after removing NaN from essential columns.")
        if verbose >= VerbosityLevel.DETAILED:
            print(f"Data NaN summary:\n{data.isna().sum()}")

        opm = deepcopy(opm) if opm else deepcopy(self.opm0)
        
        # Get default f-number if not provided
        if initial_fnumber is None:
            # Extract the default f-number from the optical model
            try:
                initial_fnumber = opm.optical_spec.pupil.value
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Using default f-number from optical model: {initial_fnumber}")
            except:
                # Default to 0.95 for the Nikkor lens if we can't extract it
                initial_fnumber = 0.95
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Could not extract f-number from optical model, using default: {initial_fnumber}")

        # Validate optimize_param
        valid_params = ["zfocus", "zfine", "fnumber", "both", "zfocus+fnumber", "zfine+fnumber", "all"]
        if optimize_param not in valid_params:
            raise ValueError(f"optimize_param must be one of {valid_params}")

        # Determine which parameters to optimize
        optimize_zfocus = optimize_param in ["zfocus", "both", "zfocus+fnumber", "all"]
        optimize_zfine = optimize_param in ["zfine", "both", "zfine+fnumber", "all"]
        optimize_fnumber = optimize_param in ["fnumber", "zfocus+fnumber", "zfine+fnumber", "all"]

        # Objective function factory with support for f-number
        def create_objective(param_name: str, fixed_params: dict):
            best_result = {
                'z': fixed_params.get('zfocus', initial_zfocus) if param_name == 'zfocus' else 
                    fixed_params.get('zfine', initial_zfine) if param_name == 'zfine' else
                    fixed_params.get('fnumber', initial_fnumber),
                'std': float('inf'), 
                'traced_fraction': 0.0
            }
            iteration_count = [0]

            def objective(params):
                z = params['z'].value
                
                # Create a dictionary of parameters for refocus method
                refocus_kwargs = fixed_params.copy()
                refocus_kwargs[param_name] = z
                
                try:
                    current_opm = self.refocus(opm, **refocus_kwargs)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, refocus failed: {str(e)}")
                    iteration_count[0] += 1
                    return float('inf')

                df = self.trace_rays(opm=current_opm, join=False, print_stats=False,
                                n_processes=n_processes, chunk_size=chunk_size,
                                progress_bar=False, return_df=True, verbosity=verbose)

                if df is None or df.empty:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, trace_rays returned None or empty")
                    iteration_count[0] += 1
                    return float('inf')

                if 'x2' not in df.columns or 'y2' not in df.columns:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, missing x2/y2 columns")
                    iteration_count[0] += 1
                    return float('inf')

                valid_df = df.dropna(subset=['x2', 'y2'])
                traced_count = len(valid_df)
                traced_fraction = traced_count / total_photons if total_photons > 0 else 0.0

                if traced_count < 2:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, insufficient valid data (traced: {traced_count}/{total_photons})")
                    iteration_count[0] += 1
                    return 1e6 + 1000.0 * (1.0 - traced_fraction)  # High base value to avoid NaN issues

                std_x2 = valid_df['x2'].std()
                std_y2 = valid_df['y2'].std()
                std_value = np.sqrt(std_x2**2 + std_y2**2) if not (pd.isna(std_x2) or pd.isna(std_y2)) else float('inf')

                # Objective: minimize std, heavily penalize low traced fraction
                penalty = 1000.0 * (1.0 - traced_fraction)
                objective_value = std_value + penalty if std_value != float('inf') else 1e6 + penalty

                # Different reporting for fnumber optimization
                if verbose >= VerbosityLevel.DETAILED:
                    param_str = f"f/{z:.2f}" if param_name == 'fnumber' else f"{param_name}={z:.3f}"
                    print(f"Iteration {iteration_count[0]}: {param_str}, std={std_value:.3f}, "
                        f"traced={traced_count}/{total_photons} ({traced_fraction:.2%}), penalty={penalty:.3f}, objective={objective_value:.3f}")

                if objective_value < (best_result['std'] + 1000.0 * (1.0 - best_result['traced_fraction'])):
                    best_result['z'] = z
                    best_result['std'] = std_value
                    best_result['traced_fraction'] = traced_fraction

                iteration_count[0] += 1
                return objective_value

            return objective, best_result

        results = {}
        current_zfocus = initial_zfocus
        current_zfine = initial_zfine
        current_fnumber = initial_fnumber

        # Optimize zfocus
        if optimize_zfocus:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for zfocus with initial value {current_zfocus:.3f}")
            
            fixed_params = {'zfine': current_zfine, 'fnumber': current_fnumber}
            objective_func, best_result = create_objective('zfocus', fixed_params)

            params = Parameters()
            params.add('z', value=current_zfocus, min=zfocus_min, max=zfocus_max)

            try:
                result = minimize(objective_func, params, method='nelder')  # Nelder-Mead method
                best_zfocus = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_zfocus'] = best_zfocus
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized zfocus: {best_zfocus:.3f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

                current_zfocus = best_zfocus

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of zfocus failed: {str(e)}")
                results['best_zfocus'] = current_zfocus
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        # Optimize zfine
        if optimize_zfine:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for zfine with initial value {current_zfine:.3f}")
            
            fixed_params = {'zfocus': current_zfocus, 'fnumber': current_fnumber}
            objective_func, best_result = create_objective('zfine', fixed_params)

            params = Parameters()
            params.add('z', value=current_zfine, min=zfine_min, max=zfine_max)

            try:
                result = minimize(objective_func, params, method='nelder')
                best_zfine = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_zfine'] = best_zfine
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized zfine: {best_zfine:.3f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

                current_zfine = best_zfine

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of zfine failed: {str(e)}")
                results['best_zfine'] = current_zfine
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        # Optimize fnumber
        if optimize_fnumber:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for f-number with initial value f/{current_fnumber:.2f}")
            
            fixed_params = {'zfocus': current_zfocus, 'zfine': current_zfine}
            objective_func, best_result = create_objective('fnumber', fixed_params)

            params = Parameters()
            params.add('z', value=current_fnumber, min=fnumber_min, max=fnumber_max)

            try:
                result = minimize(objective_func, params, method='nelder')
                best_fnumber = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_fnumber'] = best_fnumber
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized f-number: f/{best_fnumber:.2f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of f-number failed: {str(e)}")
                results['best_fnumber'] = current_fnumber
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        return results


    def plot(self, opm: "OpticalModel"=None, **kwargs):
        """
        Plot the lens layout

        Input:
            - opm: OpticalModel, Optional, Optical model to plot
            - **kwargs: Additional keyword arguments to pass to the plot function
        """
        opm = opm if opm else self.opm0
        layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm).plot(**kwargs)
