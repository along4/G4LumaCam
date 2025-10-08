import logging
# Suppress all INFO messages globally
logging.disable(logging.INFO)
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec, InteractiveLayout
from rayoptics.environment import RayFanFigure, SpotDiagramFigure, Fit, open_model
from rayoptics.gui import roafile
from rayoptics.elem.elements import Element
from rayoptics.raytr.trace import apply_paraxial_vignetting, trace_base
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from pathlib import Path
from multiprocessing import Pool
from functools import partial   
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from rayoptics.raytr import analyses
from lmfit import Parameters, minimize, MinimizerException
from copy import deepcopy
import glob
import importlib.resources
from enum import IntEnum
from io import StringIO
from contextlib import redirect_stdout
import tempfile
import os
import struct
import shutil

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

def _process_ray_chunk_standalone(chunk, opm_file_path, wvl_values, verbosity=0):
    """
    Process a chunk of rays using an optical model loaded from file.
    
    This standalone function is designed to be pickle-safe for multiprocessing 
    by loading the optical model from a file path rather than passing the object directly.
    
    Parameters:
    -----------
    chunk : list
        List of ray tuples: (position, direction, wavelength)
    opm_file_path : str
        Path to the optical model file (.roa)
    wvl_values : numpy.ndarray
        Wavelength specification array for the optical model
    verbosity : int
        Verbosity level for logging (0=QUIET, 1=BASIC, 2=DETAILED)
        
    Returns:
    --------
    list or None
        List of traced ray results, or None for failed traces
    """
    try:
        # Log start of chunk processing
        if verbosity >= 2:
            print(f"Worker processing chunk of {len(chunk)} rays with opm_file: {opm_file_path}")

        # Load the optical model from file
        try:
            opt_model = open_model(opm_file_path)
        except Exception as e:
            if verbosity >= 2:
                print(f"Error loading optical model from {opm_file_path}: {str(e)}")
            return [None] * len(chunk)
        
        # Set the spectral region
        try:
            opt_model.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=1)
            if verbosity >= 2:
                print(f"Spectral region set with wvl_values: {wvl_values}")
        except Exception as e:
            if verbosity >= 2:
                print(f"Error setting spectral region with wvl_values={wvl_values}: {str(e)}")
            return [None] * len(chunk)
        
        # Trace the rays
        try:
            result = analyses.trace_list_of_rays(
                opt_model,
                chunk,
                output_filter="last",
                rayerr_filter="summary"
            )
            if verbosity >= 2:
                print(f"Successfully traced {len(result)} rays")
            return result
        except Exception as e:
            if verbosity >= 2:
                print(f"Error tracing rays: {str(e)}")
            return [None] * len(chunk)
            
    except Exception as e:
        if verbosity >= 2:
            print(f"Unexpected error in _process_ray_chunk_standalone: {str(e)}")
        return [None] * len(chunk)

class Lens:
    """
    Lens defining object with integrated data management.
    """
    def __init__(self, archive: str = None, data: "pd.DataFrame" = None,
                 kind: str = "nikkor_58mm", focus: float = None, zmx_file: str = None,
                 focus_gaps: List[Tuple[int, float]] = None, dist_from_obj: float = None,
                 gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0, fnumber: float = 8.0):
        """
        Initialize a Lens object with optical model and data management.

        Args:
            archive (str, optional): Directory path for saving results.
            data (pd.DataFrame, optional): Optical photon data table.
            kind (str, optional): Lens type ('nikkor_58mm', 'microscope', 'zmx_file'). Defaults to 'nikkor_58mm'.
            focus (float, optional): Initial focus adjustment in mm relative to default settings.
            zmx_file (str, optional): Path to .zmx file for custom lens (required when kind='zmx_file').
            focus_gaps (List[Tuple[int, float]], optional): List of (gap_index, scaling_factor) for focus adjustment.
            dist_from_obj (float, optional): Distance from object to first lens in mm. Defaults to 35.0.
            gap_between_lenses (float, optional): Gap between lenses in mm. Defaults to 15.0.
            dist_to_screen (float, optional): Distance from last lens to screen in mm. Defaults to 20.0.
            fnumber (float, optional): F-number of the optical system. Defaults to 8.0.

        Raises:
            ValueError: If invalid lens kind, missing zmx_file for 'zmx_file', or invalid parameters.
        """
        self.kind = kind
        self.focus = focus
        self.zmx_file = zmx_file
        self.focus_gaps = focus_gaps

        # Set default parameters based on lens kind
        if kind == "nikkor_58mm":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 461.535  # Match imported model
            self.gap_between_lenses = 0.0
            self.dist_to_screen = 0.0
            self.fnumber = fnumber if fnumber != 8.0 else 0.98
            self.default_focus_gaps = [(22, 2.68)]  # Default thickness for gap 22
        elif kind == "microscope":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 41.0  # Default distance for microscope
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = [(24, None), (31, None)]  # Will be set after loading
        elif kind == "zmx_file":
            self.dist_from_obj = dist_from_obj
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = focus_gaps or []
        else:
            raise ValueError(f"Unknown lens kind: {kind}, supported lenses are ['nikkor_58mm', 'microscope', 'zmx_file']")

        # Validate inputs
        if kind == "zmx_file" and zmx_file is None:
            raise ValueError("zmx_file must be provided when kind='zmx_file'")
        if kind == "zmx_file" and focus_gaps is None:
            print("Warning: focus_gaps not provided for zmx_file; zfine will have no effect unless specified")

        if archive is not None:
            self.archive = Path(archive)
            self.archive.mkdir(parents=True, exist_ok=True)

            sim_photons_dir = self.archive / "SimPhotons"
            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))

            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data"):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")
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

        # Initialize optical models
        self.opm0 = None
        self.opm = None
        if self.kind == "nikkor_58mm":
            self.opm0 = self.nikkor_58mm(dist_from_obj=self.dist_from_obj, fnumber=self.fnumber, save=False)
            self.opm = deepcopy(self.opm0)
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        elif self.kind == "microscope":
            self.opm0 = self.microscope_nikor_80_200mm_canon_50mm(focus=focus or 0.0, save=False)
            self.opm = deepcopy(self.opm0)
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        elif self.kind == "zmx_file":
            self.opm0 = self.load_zmx_lens(zmx_file, focus=focus, save=False)
            self.opm = deepcopy(self.opm0)
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        else:
            raise ValueError(f"Unknown lens kind: {self.kind}")

    def get_first_order_parameters(self, opm: "OpticalModel" = None) -> pd.DataFrame:
        """
        Calculate first-order optical parameters and return them as a DataFrame.

        Args:
            opm (OpticalModel, optional): Optical model to analyze. Defaults to self.opm0.

        Returns:
            pd.DataFrame: DataFrame with first-order parameters and user-friendly names.

        Raises:
            RuntimeError: If parameters cannot be retrieved.
        """
        if opm is None:
            opm = self.opm0
        pm = opm['parax_model']

        output = StringIO()
        with redirect_stdout(output):
            pm.first_order_data()
        output_str = output.getvalue()
        
        fod = {}
        multi_word_keys = ['pp sep', 'na obj', 'n obj', 'na img', 'n img', 'optical invariant']
        lines = output_str.strip().split('\n')
        for line in lines:
            try:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                key, value = parts
                for mw_key in multi_word_keys:
                    if line.startswith(mw_key):
                        key = mw_key
                        value = line[len(mw_key):].strip()
                        break
                try:
                    fod[key] = float(value)
                except ValueError:
                    fod[key] = value
            except ValueError:
                continue
        
        if not fod:
            fod = {}
            try:
                fod['efl'] = pm.efl if hasattr(pm, 'efl') else float('nan')
                fod['f'] = pm.f if hasattr(pm, 'f') else float('nan')
                fod['f\''] = pm.f_prime if hasattr(pm, 'f_prime') else float('nan')
                fod['ffl'] = pm.ffl if hasattr(pm, 'ffl') else float('nan')
                fod['pp1'] = pm.pp1 if hasattr(pm, 'pp1') else float('nan')
                fod['bfl'] = pm.bfl if hasattr(pm, 'bfl') else float('nan')
                fod['ppk'] = pm.ppk if hasattr(pm, 'ppk') else float('nan')
                fod['pp sep'] = pm.pp_sep if hasattr(pm, 'pp_sep') else float('nan')
                fod['f/#'] = pm.f_number if hasattr(pm, 'f_number') else opm['optical_spec'].pupil.value
                fod['m'] = pm.magnification if hasattr(pm, 'magnification') else float('nan')
                fod['red'] = pm.reduction if hasattr(pm, 'reduction') else float('nan')
                fod['obj_dist'] = pm.obj_dist if hasattr(pm, 'obj_dist') else opm.seq_model.gaps[0].thi
                fod['obj_ang'] = pm.obj_angle if hasattr(pm, 'obj_angle') else opm['optical_spec'].field_of_view.flds[-1]
                fod['enp_dist'] = pm.enp_dist if hasattr(pm, 'enp_dist') else float('nan')
                fod['enp_radius'] = pm.enp_radius if hasattr(pm, 'enp_radius') else float('nan')
                fod['na obj'] = pm.na_obj if hasattr(pm, 'na_obj') else float('nan')
                fod['n obj'] = pm.n_obj if hasattr(pm, 'n_obj') else 1.0
                fod['img_dist'] = pm.img_dist if hasattr(pm, 'img_dist') else float('nan')
                fod['img_ht'] = pm.img_height if hasattr(pm, 'img_height') else float('nan')
                fod['exp_dist'] = pm.exp_dist if hasattr(pm, 'exp_dist') else float('nan')
                fod['exp_radius'] = pm.exp_radius if hasattr(pm, 'exp_radius') else float('nan')
                fod['na img'] = pm.na_img if hasattr(pm, 'na_img') else float('nan')
                fod['n img'] = pm.n_img if hasattr(pm, 'n_img') else 1.0
                fod['optical invariant'] = pm.opt_inv if hasattr(pm, 'opt_inv') else float('nan')
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve first-order parameters: {e}")

        param_names = {
            'efl': 'Effective Focal Length (mm)',
            'f': 'Focal Length (mm)',
            'f\'': 'Back Focal Length (mm)',
            'ffl': 'Front Focal Length (mm)',
            'pp1': 'Front Principal Point (mm)',
            'bfl': 'Back Focal Length to Image (mm)',
            'ppk': 'Back Principal Point (mm)',
            'pp sep': 'Principal Plane Separation (mm)',
            'f/#': 'F-Number',
            'm': 'Magnification',
            'red': 'Reduction Ratio',
            'obj_dist': 'Object Distance (mm)',
            'obj_ang': 'Object Field Angle (degrees)',
            'enp_dist': 'Entrance Pupil Distance (mm)',
            'enp_radius': 'Entrance Pupil Radius (mm)',
            'na obj': 'Object Numerical Aperture',
            'n obj': 'Object Space Refractive Index',
            'img_dist': 'Image Distance (mm)',
            'img_ht': 'Image Height (mm)',
            'exp_dist': 'Exit Pupil Distance (mm)',
            'exp_radius': 'Exit Pupil Radius (mm)',
            'na img': 'Image Numerical Aperture',
            'n img': 'Image Space Refrictive Index',
            'optical invariant': 'Optical Invariant'
        }
        
        df = pd.DataFrame.from_dict(fod, orient='index', columns=['Value'])
        df['Original Name'] = df.index
        df.index = [param_names.get(idx, idx) for idx in df.index]
        df = df[['Original Name', 'Value']]
        return df

    def load_zmx_lens(self, zmx_file: str, focus: float = None, dist_from_obj: float = None,
                      gap_between_lenses: float = None, dist_to_screen: float = None,
                      fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Load a lens from a .zmx file.

        Args:
            zmx_file (str): Path to the .zmx file.
            focus (float, optional): Initial focus adjustment in mm relative to default settings.
            dist_from_obj (float, optional): Distance from object to first lens in mm.
            gap_between_lenses (float, optional): Gap between lenses in mm.
            dist_to_screen (float, optional): Distance from last lens to screen in mm.
            fnumber (float, optional): F-number of the optical system.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The loaded optical model.

        Raises:
            FileNotFoundError: If the .zmx file does not exist.
        """
        if not Path(zmx_file).exists():
            raise FileNotFoundError(f".zmx file not found: {zmx_file}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = f'Custom Lens from {Path(zmx_file).name}'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True
        sm.gaps[0].thi = dist_from_obj if dist_from_obj is not None else self.dist_from_obj
        osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber if fnumber is not None else self.fnumber)
        sm.do_apertures = False
        opm.add_from_file(zmx_file, t=gap_between_lenses if gap_between_lenses is not None else self.gap_between_lenses)
        if dist_to_screen is not None:
            sm.gaps[-1].thi = dist_to_screen
        elif self.dist_to_screen != 0.0:
            sm.gaps[-1].thi = self.dist_to_screen
        opm.update_model()
        
        if focus is not None and self.focus_gaps is not None:
            opm = self.refocus(opm=opm, zfine=focus, save=False)
        
        if save:
            output_path = self.archive / f"Custom_Lens_{Path(zmx_file).stem}.roa"
            opm.save_model(str(output_path))
        return opm

    def microscope_nikor_80_200mm_canon_50mm(self, focus: float = 0.0, dist_from_obj: float = 41.0,
                                             gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0,
                                             fnumber: float = 8.0, save: bool = False) -> OpticalModel:
        """
        Create a microscope lens model with Nikkor 80-200mm f/2.8 and flipped Canon 50mm f/1.8 lenses.

        Args:
            focus (float): Focus adjustment in mm relative to default settings (gap 24 increases, gap 31 decreases). Defaults to 0.0.
            dist_from_obj (float): Distance from object to first lens in mm. Defaults to 35.0.
            gap_between_lenses (float): Gap between the two lenses in mm. Defaults to 15.0.
            dist_to_screen (float): Distance from second lens to screen in mm. Defaults to 20.0.
            fnumber (float): F-number of the optical system. Defaults to 8.0.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The configured microscope optical model.

        Raises:
            ValueError: If parameters are invalid.
            FileNotFoundError: If .zmx files are not found.
        """
        if dist_from_obj <= 0:
            raise ValueError(f"dist_from_obj must be positive, got {dist_from_obj}")
        if gap_between_lenses < 0:
            raise ValueError(f"gap_between_lenses cannot be negative, got {gap_between_lenses}")
        if dist_to_screen < 0:
            raise ValueError(f"dist_to_screen cannot be negative, got {dist_to_screen}")
        if fnumber <= 0:
            raise ValueError(f"fnumber must be positive, got {fnumber}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = 'Microscope Lens Model'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True

        sm.gaps[0].thi = dist_from_obj
        osp.pupil = PupilSpec(osp, key=['object', 'f/#'], value=fnumber)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 1])  # Set field of view
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        sm.do_apertures = False
        opm.update_model()

        package = 'lumacam.data'
        zmx_files = [
            'JP1985-040604_Example01P_50mm_1.2f.zmx',
            'JP2000-019398_Example01_Tale67_80_200_AF-S_2.4f.zmx',
        ]

        with importlib.resources.as_file(importlib.resources.files(package).joinpath(zmx_files[0])) as zmx_path:
            if not zmx_path.exists():
                raise FileNotFoundError(f".zmx file not found: {zmx_path}")
            opm.add_from_file(str(zmx_path), t=gap_between_lenses)

        with importlib.resources.as_file(importlib.resources.files(package).joinpath(zmx_files[1])) as zmx_path:
            if not zmx_path.exists():
                raise FileNotFoundError(f".zmx file not found: {zmx_path}")
            opm.add_from_file(str(zmx_path), t=dist_to_screen)

        opm.flip(1, 15)
        
        # Store default gap thicknesses for microscope
        self.default_focus_gaps = [(24, sm.gaps[24].thi), (31, sm.gaps[31].thi)]
        opm = self.refocus(opm=opm, zfine=focus, save=False)
        opm.update_model()
        self.opm0 = deepcopy(opm)
        

        if save:
            output_path = self.archive / "Microscope_Lens.roa"
            opm.save_model(str(output_path))
        return opm

    def nikkor_58mm(self, dist_from_obj: float = 461.535, fnumber: float = 0.98, save: bool = False) -> OpticalModel:
        """
        Create a Nikkor 58mm f/0.95 lens model from a .zmx file, correcting specific thicknesses to match the original model.

        Args:
            dist_from_obj (float): Distance from object to first lens in mm. Defaults to 461.535.
            fnumber (float): F-number of the optical system. Defaults to 0.98.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The configured Nikkor 58mm optical model.

        Raises:
            FileNotFoundError: If the .zmx file is not found.
            ValueError: If the model has insufficient surfaces for correction.
        """
        zmx_path = str(importlib.resources.files('lumacam.data').joinpath('WO2019-229849_Example01P.zmx'))
        if not Path(zmx_path).exists():
            raise FileNotFoundError(f".zmx file not found: {zmx_path}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = 'WO2019-229849 Example 1 (Nikkor Z 58mm f/0.95 S)'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True

        # Load the .zmx file
        sm.gaps[0].thi = dist_from_obj
        osp.pupil = PupilSpec(osp, key=['object', 'f/#'], value=fnumber)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 60])
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        sm.do_apertures = False
        opm.add_from_file(zmx_path)
        opm.update_model()

        # Correct specific thicknesses
        if len(sm.gaps) <= 30:
            raise ValueError(f"Insufficient gaps in .zmx file: {len(sm.gaps)} found, expected at least 31")

        # Correct surface 22 thickness (from 21.2900 mm to 2.68000 mm)
        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            sm.gaps[22].thi = 2.68

        # Correct surface 30 thickness (from 0.00000 mm to 1.00000 mm)
        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            sm.gaps[30].thi = 1.0

        opm.update_model()
        apply_paraxial_vignetting(opm)

        # Verify corrections
        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            print(f"Warning: Surface 22 thickness {sm.gaps[22].thi:.6f} does not match expected 2.68000 mm")
        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            print(f"Warning: Surface 30 thickness {sm.gaps[30].thi:.6f} does not match expected 1.00000 mm")

        if save:
            output_path = self.archive / "Nikkor_58mm.roa"
            opm.save_model(str(output_path))

        return opm

    def refocus(self, opm: "OpticalModel" = None, zscan: float = 0, zfine: float = 0, fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Refocus the lens by adjusting gaps relative to default settings.

        Args:
            opm (OpticalModel, optional): Optical model to refocus. Defaults to self.opm0.
            zscan (float): Distance to move the lens assembly in mm relative to default object distance. Defaults to 0.
            zfine (float): Focus adjustment in mm relative to default gap thicknesses (for microscope, gap 24 increases, gap 31 decreases). Defaults to 0.
            fnumber (float, optional): New f-number for the lens.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The refocused optical model.

        Raises:
            ValueError: If lens kind is unsupported or gap indices are invalid.
        """
        opm = deepcopy(self.opm0) if opm is None else deepcopy(opm)
        sm = opm.seq_model
        osp = opm.optical_spec
        
        if self.kind == "nikkor_58mm":
            if not self.default_focus_gaps:
                raise ValueError("Default focus gaps not set for nikkor_58mm")
            gap_index, default_thi = self.default_focus_gaps[0]
            if gap_index >= len(sm.gaps):
                raise ValueError(f"Invalid gap index {gap_index} for nikkor_58mm lens")
            if zfine != 0:
                new_thi = default_thi + zfine
                sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        elif self.kind == "microscope":
            if len(self.default_focus_gaps) != 2:
                raise ValueError("Default focus gaps not set correctly for microscope")
            if zfine != 0:
                gap_index_24, default_thi_24 = self.default_focus_gaps[0]
                if gap_index_24 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_24} for microscope lens")
                if default_thi_24 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_24}")
                new_thi_24 = default_thi_24 + zfine
                sm.gaps[gap_index_24].thi = new_thi_24

                gap_index_31, default_thi_31 = self.default_focus_gaps[1]
                if gap_index_31 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_31} for microscope lens")
                if default_thi_31 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_31}")
                new_thi_31 = default_thi_31 - zfine
                sm.gaps[gap_index_31].thi = new_thi_31
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        elif self.kind == "zmx_file":
            if zfine != 0 and self.focus_gaps is not None:
                for gap_index, scaling_factor in self.focus_gaps:
                    if gap_index >= len(sm.gaps):
                        raise ValueError(f"Invalid gap index {gap_index} for zmx_file lens")
                    default_thi = sm.gaps[gap_index].thi
                    new_thi = default_thi + zfine * scaling_factor
                    sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        else:
            raise ValueError(f"Unsupported lens kind: {self.kind}")
        
        if fnumber is not None:
            osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber)
        
        sm.do_apertures = False
        opm.update_model()
        apply_paraxial_vignetting(opm)
        
        self.opm = opm
        
        if save:
            fnumber_str = f"_f{fnumber:.2f}" if fnumber is not None else ""
            save_path = self.archive / f"refocus_zscan_{zscan}_zfine_{zfine}{fnumber_str}.roa"
            opm.save_model(save_path)
        
        return opm

    def _chunk_rays(self, rays, chunk_size):
        """
        Split rays into chunks for parallel processing while preserving ray identifiers.
        
        Parameters:
        -----------
        rays : list
            List of ray tuples: (position, direction, wavelength)
        chunk_size : int
            Number of rays per chunk
            
        Returns:
        --------
        list
            List of chunks, where each chunk is a list of ray tuples
        """
        return [rays[i:i+chunk_size] for i in range(0, len(rays), chunk_size)]


    def trace_rays(self, opm=None, opm_file=None, zscan=0, zfine=0, fnumber=None,
                join=False, print_stats=False, n_processes=None, chunk_size=1000, 
                progress_bar=True, timeout=3600, return_df=False, 
                verbosity=VerbosityLevel.BASIC, deadtime=None, blob=0.0, 
                split_method="auto"):
        """
        Trace rays from simulation data files and save processed results, optionally applying pixel saturation and blob effect.

        This method processes ray data from CSV files in the 'SimPhotons' directory using either
        a provided optical model, a saved optical model file, or by creating a refocused version
        of the default optical model. If deadtime or blob is provided, it applies the saturate_photons
        method with output_format="photons" and saves the results directly to the 'TracedPhotons'
        directory, including photon_count, time_diff, nz, pz, and pulse_id columns. It then calls _write_tpx3
        to generate TPX3 files, reading trigger times from corresponding TriggerTimes CSV files (converting ns to ps).
        Otherwise, it saves raw traced results to 'TracedPhotons' with nz, pz, and pulse_id columns.

        Parameters:
        -----------
        opm : OpticalModel, optional
            Custom optical model to use. If provided, will be saved to a temporary file
            for multiprocessing compatibility. Cannot be used with opm_file.
        opm_file : str or Path, optional  
            Path to a saved optical model file (.roa). If provided, this model will be
            loaded in each worker process. Cannot be used with opm.
        zscan : float, default 0
            Distance to move the lens assembly in mm relative to default object distance.
            Only used if neither opm nor opm_file is provided.
        zfine : float, default 0
            Focus adjustment in mm relative to default gap thicknesses. Only used if 
            neither opm nor opm_file is provided.
        fnumber : float, optional
            New f-number for the lens. Applied to refocused model if neither opm nor
            opm_file is provided.
        join : bool, default False
            If True, concatenates original simulation data with traced results.
            If False, returns only traced positions and identifiers.
        print_stats : bool, default False
            If True, prints detailed tracing statistics for each processed file
        n_processes : int, optional
            Number of processes for parallel execution. If None, uses CPU count.
            Set to 1 to disable multiprocessing for debugging.
        chunk_size : int, default 1000
            Number of rays per processing chunk. Larger chunks use more memory but
            may be more efficient. Smaller chunks provide better progress tracking.
        progress_bar : bool, default True
            If True, displays progress bars during file and chunk processing
        timeout : int, default 3600
            Maximum time in seconds for processing each file (currently unused)
        return_df : bool, default False
            If True, returns a combined DataFrame of all processed files.
            If False, returns None (files are still saved to disk).
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail level:
            - QUIET: Only essential error messages
            - BASIC: Progress bars + basic file info + statistics
            - DETAILED: All available information including warnings
        deadtime : float, optional
            Deadtime in nanoseconds for pixel saturation. If provided, applies
            saturate_photons with output_format="photons" and saves results to
            'TracedPhotons', followed by _write_tpx3.
        blob : float, default 0.0
            Interaction radius in pixels for photon hits. If > 0, each photon hit affects
            all pixels within this radius. If provided, applies saturate_photons and
            triggers _write_tpx3.
        split_method : str, default "auto"
            TPX3 file splitting strategy (only used when deadtime or blob is provided):
            - "auto": Groups neutron events to minimize file count (default)
            - "event": Creates one TPX3 file per neutron_id for event-by-event analysis

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True, 
            otherwise None. Each row represents a traced ray.
        
        Raises:
        -------
        ValueError: If both opm and opm_file are provided, or if parameters are
                    invalid (e.g., negative deadtime or blob).
        FileNotFoundError: If opm_file does not exist or if no valid simulation
                            data files are found.
        RuntimeError: If tracing fails for a file.
        """
        # Validate input parameters
        if opm is not None and opm_file is not None:
            raise ValueError("Cannot specify both 'opm' and 'opm_file' parameters. Choose one.")
        
        if opm_file is not None and not Path(opm_file).exists():
            raise FileNotFoundError(f"Optical model file not found: {opm_file}")
        
        if deadtime is not None and deadtime <= 0:
            raise ValueError(f"deadtime must be positive, got {deadtime}")
        
        if blob < 0:
            raise ValueError(f"blob must be non-negative, got {blob}")

        # Set up directories
        sim_photons_dir = self.archive / "SimPhotons"
        traced_photons_dir = self.archive / "TracedPhotons"
        trigger_times_dir = self.archive / "TriggerTimes"
        traced_photons_dir.mkdir(parents=True, exist_ok=True)

        # Find all non-empty sim_data_*.csv files
        csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
        valid_files = [f for f in csv_files if f.stat().st_size > 100]

        if not valid_files:
            if verbosity > VerbosityLevel.QUIET:
                print("No valid simulation data files found in 'SimPhotons' directory.")
                print(f"Searched in: {sim_photons_dir}")
                print("Expected files matching pattern: sim_data_*.csv")
            return None

        if verbosity > VerbosityLevel.QUIET:
            print(f"Found {len(valid_files)} valid simulation files to process")

        # Handle optical model setup
        temp_opm_file = None
        opm_file_path = None
        
        try:
            if opm is not None:
                # Save provided OPM to temporary file for multiprocessing
                temp_dir = self.archive / "TempOpm"
                temp_dir.mkdir(exist_ok=True)
                temp_opm_file = temp_dir / "temp_opm_for_tracing.roa"
                opm.save_model(str(temp_opm_file))
                opm_file_path = str(temp_opm_file)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Saved temporary OPM to: {temp_opm_file}")
                    
            elif opm_file is not None:
                # Use provided file path
                opm_file_path = str(opm_file)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Using OPM file: {opm_file_path}")
                    
            else:
                # Create refocused OPM and save to temporary file
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Creating refocused OPM with zscan={zscan}, zfine={zfine}, fnumber={fnumber}")
                
                refocused_opm = self.refocus(zscan=zscan, zfine=zfine, fnumber=fnumber, save=False)
                
                temp_dir = self.archive / "TempOpm" 
                temp_dir.mkdir(exist_ok=True)
                filename = f"temp_refocus_zscan_{zscan}_zfine_{zfine}"
                if fnumber is not None:
                    filename += f"_f{fnumber:.2f}"
                filename += ".roa"
                temp_opm_file = temp_dir / filename
                refocused_opm.save_model(str(temp_opm_file))
                opm_file_path = str(temp_opm_file)
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Saved refocused OPM to: {temp_opm_file}")

            all_results = []

            # Progress bar for file processing
            file_desc = f"Processing {len(valid_files)} files"
            file_iter = tqdm(valid_files, desc=file_desc, 
                            disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
            
            clean = True  # clean flag to erase previous files only once
            for csv_file in file_iter:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"\nProcessing file: {csv_file.name}")

                # Load and validate data
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    if verbosity > VerbosityLevel.QUIET:
                        print(f"Error reading {csv_file.name}: {str(e)}")
                    continue
                    
                if df.empty:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Skipping empty file: {csv_file.name}")
                    continue

                # Validate required columns
                required_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'wavelength']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    if verbosity > VerbosityLevel.QUIET:
                        print(f"Skipping {csv_file.name}: missing columns {missing_cols}")
                    continue

                # Check for pulse_id when split_method="event"
                if split_method == "event" and 'pulse_id' not in df.columns:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Warning: {csv_file.name} missing pulse_id column required for split_method='event'")

                # Validate nz and pz columns
                nz_pz_cols = ['nz', 'pz']
                missing_nz_pz = [col for col in nz_pz_cols if col not in df.columns]
                if missing_nz_pz:
                    if verbosity > VerbosityLevel.QUIET:
                        print(f"Warning: {csv_file.name} missing columns {missing_nz_pz}. Setting nz and pz to NaN.")
                    for col in missing_nz_pz:
                        df[col] = np.nan

                # Get wavelengths for the optical model
                wvl = df["wavelength"].value_counts().to_frame().reset_index()
                wvl["count"] = 1
                wvl_values = wvl.values

                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Rays to process: {len(df)}")
                    print(f"  Unique wavelengths: {len(wvl_values)}")
                    print(f"  Chunk size: {chunk_size}")
                    print(f"  Processes: {n_processes or 'auto'}")

                # Convert DataFrame to ray format
                rays = [
                    (np.array([row.x, row.y, row.z], dtype=np.float64),
                    np.array([row.dx, row.dy, row.dz], dtype=np.float64),
                    np.array([row.wavelength], dtype=np.float64))
                    for row in df.itertuples()
                ]

                # Split rays into chunks
                chunks = self._chunk_rays(rays, chunk_size)
                index_chunks = [list(range(i, min(i + chunk_size, len(rays)))) for i in range(0, len(rays), chunk_size)]
                
                rays = None  # Clear memory

                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Created {len(chunks)} chunks for processing")

                # Process chunks in parallel
                process_chunk = partial(
                    _process_ray_chunk_standalone,
                    opm_file_path=opm_file_path,
                    wvl_values=wvl_values,
                    verbosity=verbosity
                )
                
                try:
                    if n_processes == 1:
                        # Sequential processing for debugging
                        if verbosity >= VerbosityLevel.DETAILED:
                            print("  Using sequential processing")
                        results_with_indices = []
                        chunk_iter = tqdm(enumerate(zip(chunks, index_chunks)), 
                                        total=len(chunks),
                                        desc=f"Tracing rays ({csv_file.name})",
                                        disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
                        
                        for chunk_idx, (chunk, indices) in chunk_iter:
                            chunk_result = process_chunk(chunk)
                            results_with_indices.extend(self._align_chunk_results(
                                chunk_result, indices, chunk_idx, verbosity))
                    else:
                        # Parallel processing
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"  Using parallel processing with {n_processes or 'auto'} processes")
                        
                        with Pool(processes=n_processes) as pool:
                            results_with_indices = []
                            chunk_iter = tqdm(
                                enumerate(zip(pool.imap(process_chunk, chunks), index_chunks)),
                                total=len(chunks),
                                desc=f"Tracing rays ({csv_file.name})",
                                disable=not progress_bar or verbosity == VerbosityLevel.QUIET
                            )
                            
                            for chunk_idx, (chunk_result, indices) in chunk_iter:
                                results_with_indices.extend(self._align_chunk_results(
                                    chunk_result, indices, chunk_idx, verbosity))
                            
                            pool.close()
                            pool.join()
                            
                except Exception as e:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Error in processing {csv_file.name}: {str(e)}")
                        print("Consider using n_processes=1 for debugging")
                    raise

                # Create result DataFrame from processed chunks
                result_df = self._create_result_dataframe(results_with_indices, df, join, verbosity)

                # Preserve original timing and IDs
                result_df["toa2"] = df["toa"].values if "toa" in df.columns else np.nan

                # Explicitly copy ID columns to ensure alignment
                for id_col in ["id", "neutron_id", "nz", "pz", "pulse_id"]:  # Added pulse_id here
                    if id_col in df.columns:
                        result_df[id_col] = df[id_col].values

                # Verify alignment by checking row count
                if len(result_df) != len(df):
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"  ERROR: Row count mismatch after tracing. Expected {len(df)}, got {len(result_df)}")
                    continue

                # Apply saturation if deadtime or blob is provided
                if deadtime is not None or blob > 0:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Applying saturation with deadtime={deadtime} ns, blob={blob} pixels, decay_time=100ns")
                    
                    # Save important columns before saturation
                    pre_saturation_df = result_df.copy()
                    
                    # Ensure required columns for saturation
                    required_cols = ['x2', 'y2', 'z2', 'toa2', 'id', 'neutron_id']
                    missing_cols = [col for col in required_cols if col not in result_df.columns]
                    if missing_cols:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"Cannot apply saturation to {csv_file.name}: missing columns {missing_cols}")
                        continue
                    
                    # Call saturate_photons with output_format="photons" and decay_time
                    result_df = self.saturate_photons(
                        data=result_df,
                        deadtime=deadtime,
                        blob=blob,
                        output_format="photons",
                        min_tot=20.0,
                        pixel_size=None,
                        decay_time=100.0,
                        verbosity=verbosity
                    )
                    
                    if result_df is None or result_df.empty:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"  Saturation produced no results for {csv_file.name}")
                        continue
                    
                    # Add pulse_id by matching on original photon id
                    if "pulse_id" in pre_saturation_df.columns:
                        # Create mapping from photon id to pulse_id
                        id_to_pulse = pre_saturation_df.set_index('id')['pulse_id'].to_dict()
                        # Map pulse_id using the photon id column
                        result_df['pulse_id'] = result_df['id'].map(id_to_pulse)
                        
                        # Verify mapping worked
                        if result_df['pulse_id'].isna().any():
                            if verbosity >= VerbosityLevel.DETAILED:
                                n_missing = result_df['pulse_id'].isna().sum()
                                print(f"  Warning: {n_missing} rows missing pulse_id after mapping")
                    
                    # Sort by time to restore chronological order
                    result_df = result_df.sort_values('toa2').reset_index(drop=True)
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  After saturation and sorting: {len(result_df)} rows")

                # Save results to file
                output_file = traced_photons_dir / f"traced_{csv_file.name}"
                result_df.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Saved results to: {output_file}")

                # Call _write_tpx3 if deadtime or blob is provided
                if deadtime is not None or blob > 0:
                    trigger_file = trigger_times_dir / f"trigger_{csv_file.name}"
                    trigger_times_ps = None
                    trigger_pulse_ids = None
                    if trigger_file.exists():
                        try:
                            trigger_df = pd.read_csv(trigger_file)
                            if 'trigger_time_ns' in trigger_df.columns and 'pulse_id' in trigger_df.columns:
                                trigger_times_ps = (trigger_df['trigger_time_ns'] * 1000).astype(np.int64).values
                                trigger_pulse_ids = trigger_df['pulse_id'].astype(np.int64).values
                            else:
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"Warning: {trigger_file.name} missing 'trigger_time_ns' or 'pulse_id' column")
                        except Exception as e:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Error reading trigger file {trigger_file.name}: {str(e)}")
                    else:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Warning: Trigger file {trigger_file.name} not found")
                    
                    self._write_tpx3(
                        trigger_times_ps=trigger_times_ps,
                        trigger_pulse_ids=trigger_pulse_ids,
                        traced_data=result_df,
                        chip_index=0,
                        verbosity=verbosity,
                        sensor_size=256,
                        split_method=split_method,
                        clean=clean
                    )
                    clean = False  # Only clean once

                # Print statistics if requested
                if print_stats and verbosity >= VerbosityLevel.BASIC:
                    self._print_tracing_stats(csv_file.name, df, result_df)

                if return_df:
                    all_results.append(result_df)

            # Return combined results if requested
            if return_df and all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
                return combined_df

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nProcessing complete. Results saved to: {traced_photons_dir}")
            
            return None

        finally:
            pass

    def _write_tpx3(
        self,
        trigger_times_ps: Union[List[int], np.ndarray] = None,
        trigger_pulse_ids: Union[List[int], np.ndarray] = None,
        traced_data: pd.DataFrame = None,
        chip_index: int = 0,
        verbosity: int = 1,
        sensor_size: int = 256,
        split_method: str = "auto",
        clean: bool = True
    ):
        """
        Convert traced photon data to valid TPX3 binary files.
        
        Intelligently splits large datasets with different strategies:
        - "auto": Groups neutron events together, splits only when file size limit reached
        - "event": Creates one file per neutron_id with corresponding trigger time based on pulse_id
        
        Parameters:
        -----------
        traced_data : pd.DataFrame
            DataFrame with columns: x2, y2, toa2, time_diff (or 'tot'), neutron_id, pulse_id (optional)
        trigger_times_ps : array-like, optional
            Trigger times in picoseconds
        trigger_pulse_ids : array-like, optional
            Pulse IDs corresponding to trigger_times_ps
        chip_index : int
            Chip index for TPX3 header (0-255)
        verbosity : int
            0=silent, 1=progress, 2=detailed debug
        sensor_size : int
            Sensor dimensions (default 256x256)
        split_method : str, default "auto"
            File splitting strategy:
            - "auto": Optimize file count by grouping neutrons
            - "event": One file per neutron_id for event-by-event analysis
        clean : bool
            If True, clears existing TPX3 files in output directory before writing new ones
        """
        # Constants
        TICK_NS = 1.5625
        MAX_CHUNK_BYTES = 65535
        TIMER_TICK_NS = 409.6
        PACKET_SIZE = 8
        
        def encode_gts_pair(timer_value):
            """Encode a GTS packet pair (LSB + MSB)."""
            timer_value = int(timer_value) & ((1 << 48) - 1)
            lsb_timer = timer_value & 0xFFFFFFFF
            lsb_word = (0x4 << 60) | (0x4 << 56) | (lsb_timer << 16)
            msb_timer = (timer_value >> 32) & 0xFFFF
            msb_word = (0x4 << 60) | (0x5 << 56) | (msb_timer << 16)
            return struct.pack("<Q", lsb_word) + struct.pack("<Q", msb_word)
        
        if traced_data is None or len(traced_data) == 0:
            if verbosity >= 1:
                print("No traced photon data provided")
            return
        
        df = traced_data.copy()
        
        # Sort by neutron_id and toa2 to ensure consistent event ordering
        if "neutron_id" in df.columns:
            df = df.sort_values(by=["neutron_id", "toa2"]).reset_index(drop=True)
        
        if verbosity >= 2:
            print(f"\nProcessing {len(df)} events for TPX3 export")
        
        # Validate required columns
        required = ["x2", "y2", "toa2"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            if verbosity >= 1:
                print(f"Missing required columns: {missing}")
            return
        
        # Check for TOT column
        if "time_diff" not in df.columns and "tot" not in df.columns:
            if verbosity >= 1:
                print("Missing time_diff or tot column")
            return
        
        # Warn if pulse_id is missing for split_method="event"
        if split_method == "event" and "pulse_id" not in df.columns:
            if verbosity >= 1:
                print("Warning: pulse_id column missing for split_method='event', falling back to trigger_times_ps")
        
        # Setup output directory and clear existing contents
        out_dir = self.archive / "tpx3Files"
        if out_dir.exists() and clean:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert positions to pixel coordinates
        px = ((df["x2"].astype(float) + 10.0) / 20.0 * sensor_size).to_numpy()
        py = ((df["y2"].astype(float) + 10.0) / 20.0 * sensor_size).to_numpy()
        px_i = np.clip(np.round(px).astype(np.int64), 0, sensor_size - 1)
        py_i = np.clip(np.round(py).astype(np.int64), 0, sensor_size - 1)
        
        # Extract timing
        toa_ns = df["toa2"].astype(float).to_numpy()
        tot_ns = df["time_diff"].to_numpy() if "time_diff" in df.columns else df["tot"].to_numpy()
        tot_ns = np.maximum(tot_ns.astype(float), 1.0)
        
        # Convert to TPX3 ticks
        toa_ticks = np.round(toa_ns / TICK_NS).astype(np.int64)
        tot_ticks = np.clip(np.round(tot_ns / TICK_NS).astype(np.int64), 1, 0x3FF)
        
        # Decompose ToA
        spidr_time = ((toa_ticks >> 18) & 0xFFFF).astype(np.int64)
        coarse_raw = ((toa_ticks >> 4) & 0x3FFF).astype(np.int64)
        ftoa = (15 - (toa_ticks & 0xF)).astype(np.int64)
        ftoa = np.clip(ftoa, 0, 15)
        
        # Pixel address
        pixaddr = (py_i * sensor_size + px_i).astype(np.int64)
        
        # Timer values for GTS
        timer_ticks = (toa_ticks / (TIMER_TICK_NS / TICK_NS)).astype(np.int64)
        
        if verbosity >= 2:
            print(f"  ToA range: {toa_ns.min():.2f} - {toa_ns.max():.2f} ns")
            print(f"  Pixel range: x=[{px_i.min()}, {px_i.max()}], y=[{py_i.min()}, {py_i.max()}]")
        
        # Encode all pixel packets first
        pixel_packets = []
        for j in range(len(df)):
            p = int(pixaddr[j]) & 0xFFFF
            c = int(coarse_raw[j]) & 0x3FFF
            tot = int(tot_ticks[j]) & 0x3FF
            ft = int(ftoa[j]) & 0xF
            sp = int(spidr_time[j]) & 0xFFFF
            
            pixel_word = (0xB << 60) | (p << 44) | (c << 30) | (tot << 20) | (ft << 16) | sp
            pixel_packets.append(struct.pack("<Q", pixel_word))
        
        # Determine file groups and associated trigger times
        file_groups = []
        trigger_times_per_file = []
        
        if split_method == "event" and "neutron_id" in df.columns:
            neutron_ids = df["neutron_id"].to_numpy()
            unique_neutron_ids = np.unique(neutron_ids)
            
            # Validate neutron_id uniqueness
            if len(unique_neutron_ids) < len(neutron_ids):
                if verbosity >= 2:
                    print(f"  Warning: Found duplicate neutron IDs, ensuring correct separation")
            
            pulse_ids = df["pulse_id"].to_numpy() if "pulse_id" in df.columns else None
            
            for nid in unique_neutron_ids:
                indices = np.where(neutron_ids == nid)[0]
                if len(indices) > 0:
                    trigger_time_ps = None
                    
                    if pulse_ids is not None and trigger_pulse_ids is not None and trigger_times_ps is not None:
                        # Get pulse_id for this neutron event
                        pulse_id = pulse_ids[indices[0]]  # Assume all rows for this neutron_id have the same pulse_id
                        # Validate that all rows for this neutron_id have the same pulse_id
                        if not (pulse_ids[indices] == pulse_id).all():
                            if verbosity >= 2:
                                print(f"  Warning: Multiple pulse_ids found for neutron_id {nid}, using first: {pulse_id}")
                        
                        # Look up trigger time using pulse_id
                        trigger_indices = np.where(trigger_pulse_ids == pulse_id)[0]
                        if len(trigger_indices) > 0:
                            trigger_time_ps = trigger_times_ps[trigger_indices[0]]  # Use first match
                            if len(trigger_indices) > 1 and verbosity >= 2:
                                print(f"  Warning: Multiple trigger times found for pulse_id {pulse_id}, using first: {trigger_time_ps} ps")
                            if verbosity >= 2:
                                print(f"  Found trigger time {trigger_time_ps} ps for neutron_id {nid} (pulse_id {pulse_id})")
                        else:
                            if verbosity >= 2:
                                print(f"  Warning: No trigger time found for neutron_id {nid} (pulse_id {pulse_id})")
                    
                    # Fallback to first trigger_times_ps if no pulse_id mapping is available
                    if trigger_time_ps is None and trigger_times_ps is not None and len(trigger_times_ps) > 0:
                        trigger_time_ps = trigger_times_ps[0]
                        if verbosity >= 2:
                            print(f"  Falling back to first trigger time for neutron_id {nid}: {trigger_time_ps} ps")
                    
                    file_groups.append((indices[0], indices[-1] + 1, nid))
                    trigger_times_per_file.append([trigger_time_ps] if trigger_time_ps is not None else [])
            
            if verbosity >= 2:
                print(f"  Found {len(unique_neutron_ids)} unique neutron events")
        elif "neutron_id" in df.columns:
            neutron_ids = df["neutron_id"].to_numpy()
            neutron_boundaries = [0]
            for i in range(1, len(neutron_ids)):
                if neutron_ids[i] != neutron_ids[i-1]:
                    neutron_boundaries.append(i)
            neutron_boundaries.append(len(df))
            
            if verbosity >= 2:
                print(f"  Found {len(neutron_boundaries)-1} neutron events")
            
            current_group_start = 0
            current_group_size = 16
            
            for i in range(len(neutron_boundaries) - 1):
                start_idx = neutron_boundaries[i]
                end_idx = neutron_boundaries[i + 1]
                neutron_size = (end_idx - start_idx) * PACKET_SIZE
                
                if current_group_size + neutron_size > MAX_CHUNK_BYTES:
                    if start_idx > current_group_start:
                        file_groups.append((current_group_start, start_idx, None))
                        trigger_times_per_file.append(list(trigger_times_ps) if trigger_times_ps is not None else [])
                    current_group_start = start_idx
                    current_group_size = 16 + neutron_size
                else:
                    current_group_size += neutron_size
            
            if current_group_start < len(df):
                file_groups.append((current_group_start, len(df), None))
                trigger_times_per_file.append(list(trigger_times_ps) if trigger_times_ps is not None else [])
        else:
            if verbosity >= 2:
                print("  No neutron_id column, splitting on packet boundaries")
            
            max_events_per_file = (MAX_CHUNK_BYTES - 16) // PACKET_SIZE
            for start in range(0, len(df), max_events_per_file):
                end = min(start + max_events_per_file, len(df))
                file_groups.append((start, end, None))
                trigger_times_per_file.append(list(trigger_times_ps) if trigger_times_ps is not None else [])
        
        if verbosity >= 1:
            print(f"  Splitting into {len(file_groups)} file(s)")
        
        # Determine base filename
        traced_dir = self.archive / "TracedPhotons"
        csv_files = sorted(traced_dir.glob("traced_sim_data_*.csv"))
        
        base_name = "traced_data"
        for csv_file in csv_files:
            try:
                file_df = pd.read_csv(csv_file)
                if len(file_df) == len(traced_data):
                    base_name = csv_file.stem.replace("traced_", "")
                    break
            except:
                continue
        
        # Write files
        files_written = []
        for file_idx, ((start_idx, end_idx, neutron_id), file_trigger_times) in enumerate(zip(file_groups, trigger_times_per_file)):
            # Build initial GTS pair
            initial_timer = timer_ticks[start_idx] if start_idx < len(timer_ticks) else 0
            file_content = encode_gts_pair(initial_timer)
            
            # Add TDC triggers for this file
            if file_trigger_times is not None and len(file_trigger_times) > 0:
                trigger_arr = np.asarray(file_trigger_times, dtype=float)
                trig_ticks = np.round(trigger_arr / (TICK_NS * 1000)).astype(np.int64)
                
                for t_idx, tval in enumerate(trig_ticks):
                    trig_counter = t_idx & 0xFFF
                    timestamp = tval & ((1 << 35) - 1)
                    tdc_word = (0x6 << 60) | (trig_counter << 44) | (timestamp << 9)
                    file_content += struct.pack("<Q", tdc_word)
            
            # Add pixel packets for this range
            for j in range(start_idx, end_idx):
                file_content += pixel_packets[j]
            
            # Determine filename
            if split_method == "event" and neutron_id is not None:
                out_path = out_dir / f"{base_name}_neutron{int(neutron_id):06d}.tpx3"
            elif len(file_groups) > 1:
                out_path = out_dir / f"{base_name}_part{file_idx + 1:03d}.tpx3"
            else:
                out_path = out_dir / f"{base_name}.tpx3"
            
            # Write file
            if out_path.exists():
                out_path.unlink()
            
            with open(out_path, "wb") as fh:
                file_size = len(file_content)
                header = struct.pack("<4sBBH", b"TPX3", chip_index & 0xFF, 0, file_size & 0xFFFF)
                fh.write(header)
                fh.write(file_content)
            
            files_written.append((out_path, end_idx - start_idx))
            
            if verbosity >= 2:
                neutron_info = f"neutron {neutron_id}" if neutron_id is not None else f"part {file_idx + 1}"
                trigger_info = f", {len(file_trigger_times)} triggers" if file_trigger_times is not None and len(file_trigger_times) > 0 else ", no triggers"
                print(f"  File {file_idx + 1}: {out_path.name}, {end_idx - start_idx} events, {file_size} bytes ({neutron_info}{trigger_info})")
        
        if verbosity >= 1:
            if len(files_written) == 1:
                print(f"✅ Wrote {files_written[0][0].name}: {files_written[0][1]} events")
            else:
                print(f"✅ Wrote {len(files_written)} files:")
                for fp, n_events in files_written:
                    print(f"   - {fp.name} ({n_events} events)")

    def _align_chunk_results(self, chunk_result, indices, chunk_idx, verbosity):
        """
        Align chunk processing results with original row indices.
        
        Parameters:
        -----------
        chunk_result : list or None
            Results from processing a chunk of rays
        indices : list
            Original row indices for this chunk
        chunk_idx : int
            Index of the current chunk (for error reporting)
        verbosity : VerbosityLevel
            Current verbosity level
            
        Returns:
        --------
        list
            List of (result, index) tuples properly aligned
        """
        if chunk_result is None:
            chunk_result = [None] * len(indices)
        elif len(chunk_result) != len(indices):
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"    Warning: Chunk {chunk_idx} returned {len(chunk_result)} "
                    f"results but expected {len(indices)}")
            if len(chunk_result) < len(indices):
                chunk_result = chunk_result + [None] * (len(indices) - len(chunk_result))
            else:
                chunk_result = chunk_result[:len(indices)]
        
        return list(zip(chunk_result, indices))

    def _create_result_dataframe(self, results_with_indices, original_df, join, verbosity):
        """
        Create a DataFrame from traced ray results, including nz and pz from original data.
        
        Parameters:
        -----------
        results_with_indices : list
            List of (result, original_index) tuples
        original_df : pd.DataFrame
            Original simulation data containing nz and pz
        join : bool
            Whether to join with original data
        verbosity : VerbosityLevel
            Current verbosity level
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with traced ray results, including nz and pz
        """
        results_with_indices.sort(key=lambda x: x[1])
        
        if len(results_with_indices) != len(original_df):
            if verbosity > VerbosityLevel.BASIC:
                print(f"    Warning: Result count mismatch. Expected {len(original_df)}, "
                    f"got {len(results_with_indices)}")
            
            if len(results_with_indices) < len(original_df):
                missing_indices = set(range(len(original_df))) - set(idx for _, idx in results_with_indices)
                for idx in sorted(missing_indices):
                    results_with_indices.append((None, idx))
                results_with_indices.sort(key=lambda x: x[1])
            else:
                results_with_indices = results_with_indices[:len(original_df)]
        
        processed_results = []
        for entry, row_idx in results_with_indices:
            row_data = {
                "x2": np.nan, "y2": np.nan, "z2": np.nan
            }
            if entry is not None:
                try:
                    ray, path_length, wvl = entry
                    position = ray[0]
                    row_data.update({
                        "x2": position[0], "y2": position[1], "z2": position[2]
                    })
                except Exception as e:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"    Error extracting result: {str(e)}")
            
            processed_results.append(row_data)

        result_df = pd.DataFrame(processed_results, index=[idx for _, idx in results_with_indices])
        result_df = result_df.sort_index()

        if join:
            result = pd.merge(original_df, result_df, 
                            left_index=True, right_index=True, how="left")
        else:
            result = result_df.copy()
            id_cols = ["id", "neutron_id", "nz", "pz"]
            for col in id_cols:
                if col in original_df.columns:
                    result[col] = original_df[col].values
                    
            if "toa" in original_df.columns:
                result["toa2"] = original_df["toa"].values

        return result


    def saturate_photons(self, data: pd.DataFrame = None, deadtime: float = 600.0, blob: float = 0.0, 
                        output_format: str = "tpx3", min_tot: float = 20.0, pixel_size: float = None, 
                        decay_time: float = 100.0, verbosity: VerbosityLevel = VerbosityLevel.BASIC
                        ) -> Union[pd.DataFrame, None]:
        """
        Process traced photons to simulate an image intensifier coupled to an event camera.

        Physical model:
        1. Photon hits image intensifier at position (x, y)
        2. Intensifier creates a circular blob on the camera with specified radius
        3. Each pixel within blob is activated with a time drawn from exponential decay
        4. Each pixel has independent deadtime (default 600ns)
        5. During deadtime, additional photons update TOT but not TOA
        6. TOA = time of first photon to hit that pixel
        7. TOT = time from first photon to last photon within deadtime window

        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing photon data to process. If None, loads from 'TracedPhotons' directory.
        deadtime : float, default 600.0
            Deadtime in nanoseconds for pixel saturation. During this window after first photon,
            pixel accumulates additional photons but doesn't reset.
        blob : float, default 0.0
            Blob radius in pixel units (can be float). Each photon from the intensifier triggers 
            all camera pixels within this radius. A larger blob increases the number of triggered pixels.
            Example: blob=0 → 1 pixel per photon, blob=1 → ~9 pixels, blob=2 → ~25 pixels.
        output_format : str, default "tpx3"
            Output format: "tpx3" for pixel-based output (x, y, toa, tot) or 
            "photons" for photon-averaged output with nz, pz columns.
        min_tot : float, default 20.0
            Minimum Time-Over-Threshold in nanoseconds for TPX3 format.
        pixel_size : float, optional
            Size of each pixel in mm. If None, auto-determined as sensor_range/256.
        decay_time : float, default 100.0
            Exponential decay time constant in nanoseconds for blob activation timing.
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail level.

        Returns:
        --------
        pd.DataFrame or None
            For "tpx3": columns are x, y, toa, tot (pixel positions in mm, arrival time, time-over-threshold)
            For "photons": columns are x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff, nz, pz
        """
        if output_format not in ["tpx3", "photons"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'tpx3' or 'photons'")
        if blob < 0:
            raise ValueError(f"blob must be non-negative, got {blob}")
        if deadtime is not None and deadtime <= 0:
            raise ValueError(f"deadtime must be positive, got {deadtime}")

        # Set up input data
        if data is not None:
            dfs = [(data, None)]
            save_results = False
        else:
            traced_photons_dir = self.archive / "TracedPhotons"
            saturated_photons_dir = self.archive / "SaturatedPhotons"
            sim_photons_dir = self.archive / "SimPhotons"
            saturated_photons_dir.mkdir(parents=True, exist_ok=True)

            csv_files = sorted(traced_photons_dir.glob("traced_sim_data_*.csv"))
            dfs = []
            for f in csv_files:
                if f.stat().st_size > 100:
                    sim_file = sim_photons_dir / f.name.replace("traced_", "")
                    sim_df = None
                    if sim_file.exists():
                        try:
                            sim_df = pd.read_csv(sim_file)
                            if sim_df.empty:
                                sim_df = None
                            else:
                                # Ensure nz and pz exist
                                for col in ['nz', 'pz']:
                                    if col not in sim_df.columns:
                                        if verbosity > VerbosityLevel.BASIC:
                                            print(f"Warning: SimPhotons file {sim_file.name} missing {col}. Setting to NaN.")
                                        sim_df[col] = np.nan
                        except Exception as e:
                            if verbosity > VerbosityLevel.BASIC:
                                print(f"Error reading SimPhotons file {sim_file.name}: {str(e)}")
                            sim_df = None
                    
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            dfs.append((df, sim_df))
                    except Exception as e:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error reading {f.name}: {str(e)}")
                        continue

            if not dfs:
                if verbosity > VerbosityLevel.BASIC:
                    print("No valid traced photon files found in 'TracedPhotons' directory.")
                return None
            save_results = True

        if verbosity > VerbosityLevel.BASIC and save_results:
            print(f"Found {len(dfs)} valid traced photon files to process")

        all_results = []
        file_desc = f"Processing {len(dfs)} files" if save_results else "Processing provided data"
        file_iter = tqdm(dfs, desc=file_desc, disable=not (verbosity >= VerbosityLevel.BASIC))

        for i, (df, sim_df) in enumerate(file_iter):
            file_name = f"provided_data_{i}.csv" if not save_results else Path(df.name).name
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nProcessing: {file_name}")
                print(f"  Input shape: {df.shape}")

            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping empty data: {file_name}")
                continue

            # Validate required columns
            required_cols = ['x2', 'y2', 'z2', 'toa2']
            if output_format == "photons":
                required_cols += ['id', 'neutron_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Skipping {file_name}: missing columns {missing_cols}")
                continue

            # Clean data
            df = df.dropna(subset=required_cols).reset_index(drop=True)
            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {file_name}: no valid data after removing NaNs")
                continue

            # Sort by arrival time
            if not (df['toa2'].diff().dropna() >= 0).all():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Sorting {file_name} by toa2")
                df = df.sort_values('toa2').reset_index(drop=True)

            # Align sim_df if needed
            if sim_df is not None and len(sim_df) == len(df):
                # Assume sim_df corresponds to original df order, so no reordering needed
                pass

            # Determine pixel size
            if pixel_size is None:
                x_range = df['x2'].max() - df['x2'].min() if len(df) > 0 else 20.0
                y_range = df['y2'].max() - df['y2'].min() if len(df) > 0 else 20.0
                pixel_size = max(x_range, y_range, 20.0) / 256.0
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Auto-determined pixel_size: {pixel_size:.6f} mm")

            # Extract data arrays
            x_mm = df['x2'].to_numpy()
            y_mm = df['y2'].to_numpy()
            z_mm = df['z2'].to_numpy()
            toa = df['toa2'].to_numpy()
            photon_ids = df['id'].to_numpy() if 'id' in df.columns else np.arange(len(df), dtype=np.int64)
            neutron_ids = df['neutron_id'].to_numpy() if 'neutron_id' in df.columns else np.full(len(df), -1, dtype=np.int64)
            pulse_ids = df['pulse_id'].to_numpy() if 'pulse_id' in df.columns else np.full(len(df), -1, dtype=np.int64)

            # Get nz, pz from sim_df or df
            if sim_df is not None and len(sim_df) == len(df):
                nz = sim_df['nz'].to_numpy() if 'nz' in sim_df.columns else np.full(len(df), np.nan)
                pz = sim_df['pz'].to_numpy() if 'pz' in sim_df.columns else np.full(len(df), np.nan)
            else:
                nz = df['nz'].to_numpy() if 'nz' in df.columns else np.full(len(df), np.nan)
                pz = df['pz'].to_numpy() if 'pz' in df.columns else np.full(len(df), np.nan)

            # Convert to pixel coordinates (centered at origin = 128, 128)
            px_float = (x_mm + 10.0) / pixel_size
            py_float = (y_mm + 10.0) / pixel_size

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"  Pixel coordinate range: x=[{px_float.min():.1f}, {px_float.max():.1f}], "
                    f"y=[{py_float.min():.1f}, {py_float.max():.1f}]")

            # Build pixel event list with blob effect and exponential decay
            # Structure: pixel_events[pixel_tuple] = list of (activation_time, original_photon_index)
            pixel_events = {}
            
            if blob > 0:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Applying circular blob effect with radius {blob} pixels and {decay_time}ns decay")
                
                for idx in range(len(df)):
                    px_center = px_float[idx]
                    py_center = py_float[idx]
                    toa_i = toa[idx]
                    
                    # Find all pixels within circular blob radius
                    px_min = int(np.floor(px_center - blob))
                    px_max = int(np.ceil(px_center + blob))
                    py_min = int(np.floor(py_center - blob))
                    py_max = int(np.ceil(py_center + blob))
                    
                    for px_j in range(px_min, px_max + 1):
                        for py_j in range(py_min, py_max + 1):
                            # Calculate distance from photon center to pixel center
                            pixel_center_x = px_j + 0.5
                            pixel_center_y = py_j + 0.5
                            dist = np.sqrt((px_center - pixel_center_x)**2 + (py_center - pixel_center_y)**2)
                            
                            if dist <= blob:
                                # Draw activation time from exponential decay
                                delay = np.random.exponential(decay_time)
                                activation_time = toa_i + delay
                                
                                pixel_key = (px_j, py_j)
                                if pixel_key not in pixel_events:
                                    pixel_events[pixel_key] = []
                                pixel_events[pixel_key].append((activation_time, idx))
            else:
                # No blob: each photon triggers only its pixel at original time
                for idx in range(len(df)):
                    px_j = int(np.floor(px_float[idx]))
                    py_j = int(np.floor(py_float[idx]))
                    pixel_key = (px_j, py_j)
                    if pixel_key not in pixel_events:
                        pixel_events[pixel_key] = []
                    pixel_events[pixel_key].append((toa[idx], idx))

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"  Total pixels triggered: {len(pixel_events)}")
                total_events = sum(len(events) for events in pixel_events.values())
                print(f"  Total pixel-photon pairs: {total_events}")
                if blob > 0:
                    avg_pixels_per_photon = total_events / len(df) if len(df) > 0 else 0
                    print(f"  Average pixels triggered per photon: {avg_pixels_per_photon:.1f}")

            # Process each pixel with deadtime logic
            result_rows = []
            
            for pixel_key, events in pixel_events.items():
                px_i, py_i = pixel_key
                
                # Sort events by activation time
                events.sort(key=lambda x: x[0])
                
                if deadtime is not None and deadtime > 0:
                    # Group events within deadtime windows
                    current_window_start = None
                    window_events = []
                    
                    for event_toa, photon_idx in events:
                        if current_window_start is None:
                            # Start new window
                            current_window_start = event_toa
                            window_events = [(event_toa, photon_idx)]
                        elif event_toa < current_window_start + deadtime:
                            # Add to current window
                            window_events.append((event_toa, photon_idx))
                        else:
                            # Process completed window
                            self._add_pixel_event(result_rows, px_i, py_i, window_events, df, 
                                        photon_ids, neutron_ids, pulse_ids, nz, pz,
                                        pixel_size, min_tot, output_format)
                            
                            # Start new window
                            current_window_start = event_toa
                            window_events = [(event_toa, photon_idx)]
                    
                    # Process final window
                    if window_events:
                        self._add_pixel_event(result_rows, px_i, py_i, window_events, df,
                                    photon_ids, neutron_ids, pulse_ids, nz, pz,
                                    pixel_size, min_tot, output_format)
                else:
                    # No deadtime: each event is separate
                    for event_toa, photon_idx in events:
                        self._add_pixel_event(result_rows, px_i, py_i, [(event_toa, photon_idx)], df,
                                    photon_ids, neutron_ids, pulse_ids, nz, pz,
                                    pixel_size, min_tot, output_format)

            # Create result DataFrame
            if not result_rows:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  No results after processing {file_name}")
                continue

            result_df = pd.DataFrame(result_rows)
            
            # Save results if processing files
            if save_results:
                output_file = saturated_photons_dir / f"saturated_{file_name}"
                result_df.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Saved results to: {output_file}")

            # Print stats
            if verbosity >= VerbosityLevel.BASIC:
                print(f"  Input photons: {len(df)}, Output events: {len(result_df)}")
                if blob > 0:
                    ratio = len(result_df) / len(df) if len(df) > 0 else 0
                    print(f"  Expansion ratio (blob effect): {ratio:.2f}x")
                if output_format == "photons" and 'photon_count' in result_df.columns:
                    total_photons = result_df['photon_count'].sum()
                    print(f"  Total photons in output: {total_photons}")

            all_results.append(result_df)

        # Combine results
        if all_results:
            try:
                combined_df = pd.concat(all_results, ignore_index=True)
                if verbosity > VerbosityLevel.BASIC:
                    print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
                return combined_df
            except ValueError as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error combining results: {str(e)}")
                return None

        if verbosity > VerbosityLevel.BASIC:
            print("\nNo results to return")
        return None


    def _add_pixel_event(self, result_rows, px_i, py_i, window_events, df, 
                        photon_ids, neutron_ids, pulse_ids, nz, pz,
                        pixel_size, min_tot, output_format):
        """Helper function to add a pixel event from a deadtime window.
        
        Important: Each pixel event should preserve the identity of the FIRST photon
        that activated that specific pixel, not mix IDs from different neutrons.
        """
        # Extract timing and indices
        toas = [e[0] for e in window_events]
        indices = [e[1] for e in window_events]
        
        toa_first = toas[0]
        toa_last = toas[-1]
        tot_measured = max(toa_last - toa_first, min_tot)
        photon_count = len(window_events)
        
        # CRITICAL: Use the FIRST photon's properties for this pixel event
        # This ensures we don't mix IDs from different neutron events
        first_idx = indices[0]
        
        if output_format == "tpx3":
            # Convert pixel coordinates back to mm (pixel center)
            x_mm = (px_i + 0.5) * pixel_size - 10.0
            y_mm = (py_i + 0.5) * pixel_size - 10.0
            
            result_rows.append({
                'x': x_mm,
                'y': y_mm,
                'toa': toa_first,
                'tot': tot_measured
            })
        else:  # photons format
            # Use PIXEL coordinates, not first photon coordinates
            x_mm = (px_i + 0.5) * pixel_size - 10.0
            y_mm = (py_i + 0.5) * pixel_size - 10.0
            
            result_rows.append({
                'x2': x_mm,  # Pixel center position
                'y2': y_mm,  # Pixel center position
                'z2': 0.0,   # Sensor plane
                'pixel_x': px_i,
                'pixel_y': py_i,
                'id': photon_ids[first_idx],           # From first photon to hit this pixel
                'neutron_id': neutron_ids[first_idx],   # From first photon to hit this pixel
                'pulse_id': pulse_ids[first_idx],       # From first photon to hit this pixel
                'toa2': toa_first,
                'photon_count': photon_count,
                'time_diff': tot_measured,
                'nz': nz[first_idx],                    # From first photon to hit this pixel
                'pz': pz[first_idx]                     # From first photon to hit this pixel
            })