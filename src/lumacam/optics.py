import logging
# Or more specifically, suppress all INFO messages globally
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
                   verbosity=VerbosityLevel.BASIC, deadtime=None):
        """
        Trace rays from simulation data files and save processed results, optionally applying pixel saturation.

        This method processes ray data from CSV files in the 'SimPhotons' directory using either
        a provided optical model, a saved optical model file, or by creating a refocused version
        of the default optical model. If deadtime is provided, it applies the saturate_photons
        method with output_format="photons" and saves the results directly to the 'TracedPhotons'
        directory, including photon_count, time_diff, nz, and pz columns. Otherwise, it saves raw traced
        results to 'TracedPhotons' with nz and pz columns.

        Processing Pipeline:
        1. Locates all non-empty 'sim_data_*.csv' files in 'SimPhotons' directory
        2. Converts ray data to appropriate format for optical tracing
        3. Processes rays in parallel chunks using multiprocessing
        4. If deadtime is provided, applies saturation to group photons by pixel and deadtime
        5. Saves results to 'TracedPhotons' directory with nz and pz from SimPhotons
        6. Optionally returns combined results as a DataFrame

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
            'TracedPhotons' instead of 'SaturatedPhotons'.

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True, 
            otherwise None. Each row represents a traced ray with columns:
            - Without deadtime: x2, y2, z2, id, neutron_id, toa2, nz, pz (if join=False)
            - With deadtime: x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff, nz, pz
            - Additional columns if join=True

        Raises:
        -------
        ValueError
            If both opm and opm_file are provided, or if optical model creation fails
        FileNotFoundError
            If opm_file is specified but file does not exist
        Exception
            If parallel processing or file operations fail
        """
        # Validate input parameters
        if opm is not None and opm_file is not None:
            raise ValueError("Cannot specify both 'opm' and 'opm_file' parameters. Choose one.")
        
        if opm_file is not None and not Path(opm_file).exists():
            raise FileNotFoundError(f"Optical model file not found: {opm_file}")
        
        if deadtime is not None and deadtime <= 0:
            raise ValueError(f"deadtime must be positive, got {deadtime}")

        # Set up directories
        sim_photons_dir = self.archive / "SimPhotons"
        traced_photons_dir = self.archive / "TracedPhotons"
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
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Error in processing {csv_file.name}: {str(e)}")
                        print("Consider using n_processes=1 for debugging")
                    raise

                # Create result DataFrame from processed chunks
                result_df = self._create_result_dataframe(results_with_indices, df, join, verbosity)
                result_df["toa2"] = df["toa"] if "toa" in df.columns else np.nan

                # Apply saturation if deadtime is provided
                if deadtime is not None:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Applying saturation with deadtime={deadtime} ns")
                    # Ensure required columns for saturation
                    required_cols = ['x2', 'y2', 'z2', 'toa2', 'id', 'neutron_id']
                    missing_cols = [col for col in required_cols if col not in result_df.columns]
                    if missing_cols:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"Cannot apply saturation to {csv_file.name}: missing columns {missing_cols}")
                        continue
                    # Call saturate_photons with output_format="photons"
                    result_df = self.saturate_photons(
                        data=result_df,
                        deadtime=deadtime,
                        output_format="photons",
                        min_tot=20.0,
                        pixel_size=None,
                        verbosity=verbosity
                    )
                    if result_df is None or result_df.empty:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"  Saturation produced no results for {csv_file.name}")
                        continue

                # Print statistics if requested
                if print_stats and verbosity >= VerbosityLevel.BASIC:
                    self._print_tracing_stats(csv_file.name, df, result_df)

                # Save results to file
                output_file = traced_photons_dir / f"traced_{csv_file.name}"
                result_df.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Saved results to: {output_file}")

                if return_df:
                    all_results.append(result_df)

            # Return combined results if requested
            if return_df and all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                if verbosity > VerbosityLevel.QUIET:
                    print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
                return combined_df

            if verbosity > VerbosityLevel.QUIET:
                print(f"\nProcessing complete. Results saved to: {traced_photons_dir}")
            
            return None

        finally:
            pass

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

    def _write_tpx3(
        self,
        trigger_times_ps: Union[List[int], np.ndarray] = None,
        traced_data: pd.DataFrame = None,
        chip_index: int = 0,
        verbosity: int = 1,
        sensor_size: int = 256,
    ):
        """
        Convert traced photon data to valid TPX3 binary files and write them to
        archive/Tpx3Files/*.tpx3

        - traced_data: optional in-memory DataFrame (if provided, only one file is written)
        - otherwise reads all CSVs from archive/TracedPhotons/*.csv (any name)
        - expected traced CSV columns: at least x2, y2, toa2, time_diff
        * `toa2` is interpreted as time in **nanoseconds** (ns).
        * `time_diff` is used as tot in **nanoseconds** (ns) and clipped to >= 1 ns.
        - Pixel mapping used:
            px = (x2 + 10) / 10 * 128
            py = (y2 + 10) / 10 * 128
        - Times are converted to TPX3 ticks (1 tick = 1.5625 ns).
        - Global time packet (type 0x4) is added at start of each chunk and every ~100 ns, split into LSB (0x4) and MSB (0x5) subheaders.
        - TDC triggers (0x6) are appended after initial global packet (if provided).
        - Writes one .tpx3 file per traced CSV into archive/Tpx3Files/
        - verbosity: 0 silent, 1 tqdm + summary, 2 detailed prints
        """
        # Helper constants
        TICK_NS = 1.5625  # ns per Timepix3 tick
        MAX_CHUNK_BYTES = 0xFFFF  # max bytes per chunk (65535)
        GTS_INTERVAL_PS = 100e3  # 100 ns in picoseconds
        GTS_TICK_NS = 409.6  # ns per GTS tick
        GTS_INTERVAL_TICKS = int(GTS_INTERVAL_PS / (GTS_TICK_NS * 1000))  # GTS interval in 409.6 ns ticks

        def _encode_gts_lsb(global_time_ticks):
            """Encode GTS LSB packet (subheader 0x4)."""
            lsb = (global_time_ticks & 0xFFFFFFFF)  # Lower 32 bits
            word = (0x4 << 60) | (0x4 << 56) | (lsb << 16)
            return struct.pack("<Q", word)

        def _encode_gts_msb(global_time_ticks):
            """Encode GTS MSB packet (subheader 0x5)."""
            msb = (global_time_ticks >> 32) & 0xFFFF  # Upper 16 bits
            word = (0x4 << 60) | (0x5 << 56) | (msb << 16)
            return struct.pack("<Q", word)

        # Prepare directories
        traced_dir = self.archive / "TracedPhotons"
        out_dir = self.archive / "tpx3Files"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Gather traced dataframes
        traced_files = []
        traced_dfs = []

        if traced_data is not None:
            if not isinstance(traced_data, pd.DataFrame):
                raise ValueError("traced_data must be a pandas DataFrame")
            traced_files = [None]
            traced_dfs = [traced_data.copy()]
        else:
            if not traced_dir.exists():
                if verbosity >= 1:
                    print(f"TracedPhotons directory not found: {traced_dir}")
                return
            csv_list = sorted(traced_dir.glob("*.csv"))
            for f in csv_list:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    if verbosity >= 2:
                        print(f"Failed to read {f.name}: {e}")
                    continue
                if not all(col in df.columns for col in ["x2", "y2", "toa2", "time_diff"]):
                    if verbosity >= 2:
                        print(f"Skipping {f.name}: missing required columns (need x2,y2,toa2,time_diff)")
                    continue
                traced_files.append(f)
                traced_dfs.append(df.copy())

        if len(traced_dfs) == 0:
            if verbosity >= 1:
                print("No traced photon files found or empty traced_data.")
            return

        # Process each traced dataframe
        for idx, df in enumerate(traced_dfs):
            src_name = traced_files[idx].name if traced_files[idx] is not None else "inmemory"
            if verbosity >= 2:
                print(f"Processing traced source: {src_name} (rows={len(df)})")

            # Build the x,y,toa,tot columns
            px = ((df["x2"].astype(float) + 10.0) / 10.0 * 128.0).to_numpy()
            py = ((df["y2"].astype(float) + 10.0) / 10.0 * 128.0).to_numpy()

            px_i = np.floor(px + 0.5).astype(np.int64)
            py_i = np.floor(py + 0.5).astype(np.int64)
            px_i = np.clip(px_i, 0, sensor_size - 1)
            py_i = np.clip(py_i, 0, sensor_size - 1)

            tot_ns = df["time_diff"].astype(float).to_numpy()
            tot_ns = np.where(np.isfinite(tot_ns), tot_ns, 1.0)
            tot_ns = np.maximum(tot_ns, 1.0)

            toa_ns = df["toa2"].astype(float).to_numpy()

            toa_ticks_total = np.rint(toa_ns / TICK_NS).astype(np.int64)
            tot_ticks = np.rint(tot_ns / TICK_NS).astype(np.int64)
            tot_ticks = np.maximum(tot_ticks, 1)

            n_hits = len(df)

            spidr_time = (toa_ticks_total >> 18).astype(np.int64)
            r = toa_ticks_total - (spidr_time << 18)
            r = r.astype(np.int64)
            coarse = ((r + 15) // 16).astype(np.int64)
            fine = (coarse * 16 - r).astype(np.int64)

            overflow_coarse_idx = np.where(coarse > 0x3FFF)[0]
            if overflow_coarse_idx.size > 0:
                if verbosity >= 2:
                    print(f"Warning: {overflow_coarse_idx.size} events have coarse>14bit, clamping them.")
                for i_over in overflow_coarse_idx:
                    c = 0x3FFF
                    f = int(c * 16 - r[i_over])
                    if f < 0:
                        f = 0
                    elif f > 15:
                        f = 15
                    coarse[i_over] = c
                    fine[i_over] = f

            pixaddr = (py_i * sensor_size + px_i).astype(np.int64)
            pixaddr = pixaddr & 0xFFFF
            coarse = coarse & 0x3FFF
            tot_ticks = tot_ticks & 0x3FF
            fine = fine & 0xF
            spidr_time = spidr_time & 0xFFFF

            events_bytes: List[bytes] = []

            # --- Initial GTS packet pair ---
            if n_hits > 0:
                global_ticks = int(toa_ticks_total[0] * (TICK_NS * 1000) / (GTS_TICK_NS * 1000))
            else:
                global_ticks = 0
            global_ticks &= ((1 << 48) - 1)
            events_bytes.append(_encode_gts_lsb(global_ticks))
            events_bytes.append(_encode_gts_msb(global_ticks))
            if verbosity == 2:
                print(f"[DEBUG] Initial GTS LSB: ticks={global_ticks & 0xFFFFFFFF}, word=0x{(0x4 << 60) | (0x4 << 56) | ((global_ticks & 0xFFFFFFFF) << 16):016x}")
                print(f"[DEBUG] Initial GTS MSB: ticks={(global_ticks >> 32) & 0xFFFF}, word=0x{(0x4 << 60) | (0x5 << 56) | (((global_ticks >> 32) & 0xFFFF) << 16):016x}")

            # --- TDC triggers ---
            if trigger_times_ps is not None and len(trigger_times_ps) > 0:
                trigger_arr = np.asarray(trigger_times_ps, dtype=float)
                trig_ticks = np.rint(trigger_arr / (TICK_NS * 1e3)).astype(np.int64)
                for t_idx, tval in enumerate(trig_ticks):
                    trig_counter = int(t_idx) & 0xFFF
                    tmask35 = int(tval & ((1 << 35) - 1))
                    tdc_word = (0x6 << 60) | (trig_counter << 44) | (tmask35 << 9)
                    events_bytes.append(struct.pack("<Q", int(tdc_word)))
                    if verbosity == 2 and t_idx < 3:
                        print(f"[DEBUG] TDC {t_idx}: ps={trigger_arr[t_idx]}, ticks={tval}, word=0x{tdc_word:016x}")

            # --- Pixel packets with periodic GTS ---
            it = range(n_hits)
            if verbosity >= 1:
                iterator = tqdm(it, desc="Encoding pixels", disable=(verbosity == 0))
            else:
                iterator = it

            last_gts = global_ticks

            for j in iterator:
                current_gts = int(toa_ticks_total[j] * (TICK_NS * 1000) / (GTS_TICK_NS * 1000))
                if current_gts > last_gts:
                    events_bytes.append(_encode_gts_lsb(current_gts))
                    events_bytes.append(_encode_gts_msb(current_gts))
                    last_gts = current_gts
                    if verbosity == 2 and j < 5:
                        print(f"[DEBUG] GTS LSB at event {j}: ticks={current_gts & 0xFFFFFFFF}, word=0x{(0x4 << 60) | (0x4 << 56) | ((current_gts & 0xFFFFFFFF) << 16):016x}")
                        print(f"[DEBUG] GTS MSB at event {j}: ticks={(current_gts >> 32) & 0xFFFF}, word=0x{(0x4 << 60) | (0x5 << 56) | (((current_gts >> 32) & 0xFFFF) << 16):016x}")

                p = int(pixaddr[j]) & 0xFFFF
                c = int(coarse[j]) & 0x3FFF
                to = int(tot_ticks[j]) & 0x3FF
                ft = int(fine[j]) & 0xF
                sp = int(spidr_time[j]) & 0xFFFF

                pixel_word = (
                    (0xB << 60)
                    | (p << 44)
                    | (c << 30)
                    | (to << 20)
                    | (ft << 16)
                    | sp
                )
                events_bytes.append(struct.pack("<Q", int(pixel_word)))

                if verbosity == 2 and j < 5:
                    print(f"[DEBUG] Pixel {j}: pix={p}, coarse={c}, tot={to}, ftoa={ft}, spidr={sp}, word=0x{pixel_word:016x}")

            # --- Write chunked with GTS at start of each chunk ---
            content = b"".join(events_bytes)
            total_bytes = len(content)
            offset = 0
            remaining = total_bytes
            chunk_index = 0

            base_name = traced_files[idx].stem if traced_files[idx] is not None else "traced_inmemory"
            out_path = out_dir / f"{base_name}.tpx3"

            if out_path.exists():
                out_path.unlink()

            with open(out_path, "ab") as fh:
                while remaining > 0:
                    chunk_size = min(remaining, MAX_CHUNK_BYTES - 16)  # Reserve space for GTS pair (2 x 8 bytes)
                    # Add GTS packet pair at start of each chunk
                    chunk_gts_ticks = int((toa_ticks_total[0] if n_hits > 0 else 0) * (TICK_NS * 1000) / (GTS_TICK_NS * 1000) + chunk_index * GTS_INTERVAL_TICKS)
                    chunk_gts_ticks &= ((1 << 48) - 1)
                    chunk_content = _encode_gts_lsb(chunk_gts_ticks) + _encode_gts_msb(chunk_gts_ticks) + content[offset:offset + chunk_size]
                    header = struct.pack("<4sBBH", b"TPX3", int(chip_index) & 0xFF, 0, len(chunk_content) & 0xFFFF)
                    fh.write(header)
                    fh.write(chunk_content)
                    if verbosity == 2:
                        print(f"[DEBUG] Chunk {chunk_index}: GTS ticks={chunk_gts_ticks}, chunk_size={len(chunk_content)}")
                    offset += chunk_size
                    remaining -= chunk_size
                    chunk_index += 1

            if verbosity >= 1:
                print(f"✅ Wrote {out_path} ({n_hits} hits, {total_bytes} bytes total, chunks={chunk_index})")

        if verbosity >= 1:
            print("All .tpx3 files written to:", str(out_dir))

            
    def _print_tracing_stats(self, filename, original_df, result_df):
        """
        Print statistics about ray tracing results.
        
        Parameters:
        -----------
        filename : str
            Name of the processed file
        original_df : pd.DataFrame  
            Original simulation data
        result_df : pd.DataFrame
            Traced results data
        """
        total = len(original_df)
        traced = result_df.dropna(subset=["x2"]).shape[0]
        failed = total - traced
        percentage = (traced / total) * 100 if total > 0 else 0
        
        print(f"File: {filename}")
        print(f" Original rays: {total:,}")
        print(f" Successfully traced: {traced:,} ({percentage:.1f}%)")
        print(f" Failed traces: {failed:,} ({100-percentage:.1f}%)")
        
        if traced > 0:
            x_range = result_df['x2'].max() - result_df['x2'].min()
            y_range = result_df['y2'].max() - result_df['y2'].min()
            z_range = result_df['z2'].max() - result_df['z2'].min()
            print(f"    Position ranges - X: {x_range:.3f}mm, Y: {y_range:.3f}mm, Z: {z_range:.3f}mm")

    def saturate_photons(self, data: pd.DataFrame = None, deadtime: float = 100.0, output_format: str = "tpx3",
                         min_tot: float = 20.0, pixel_size: float = None, verbosity: VerbosityLevel = VerbosityLevel.BASIC
                         ) -> Union[pd.DataFrame, None]:
        """
        Process traced photons to simulate an event camera with pixel saturation within a specified deadtime.

        This method processes photon data, either from the provided DataFrame or from CSV files in the
        'TracedPhotons' directory, grouping photons that arrive at the same integer pixel (256x256 grid)
        within the specified deadtime. It checks if the input data is sorted by toa2 and sorts if necessary
        to ensure correct grouping. Depending on the output_format, it generates either a TPX3-like output
        or a photon-averaged output, and saves results to the 'SaturatedPhotons' subfolder as CSV files.
        For output_format="photons", includes nz and pz columns from the corresponding SimPhotons file
        or the input DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing photon data to process. If None, loads from 'TracedPhotons' directory.
        deadtime : float, default 100.0
            Deadtime in nanoseconds for pixel saturation.
        output_format : str, default "tpx3"
            Output format: "tpx3" for pixel-based output with TOA and TOT, or "photons" for averaged photon positions
            with additional columns for photon count, time difference, nz, and pz.
        min_tot : float, default 20.0
            Minimum Time-Over-Threshold (TOT) in nanoseconds for TPX3 format.
        pixel_size : float, optional
            Size of each pixel in mm. If None, automatically determined from the data range.
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail level:
            - QUIET: Only essential error messages
            - BASIC: Progress bars + basic file info
            - DETAILED: All available information including warnings

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if data is provided or files are processed,
            otherwise None. For "tpx3" format, columns are: x, y, toa, tot
            For "photons" format, columns are: x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff, nz, pz

        Raises:
        -------
        ValueError
            If output_format is invalid or required columns are missing
        FileNotFoundError
            If no valid traced photon files are found when data is None
        """
        if output_format not in ["tpx3", "photons"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'tpx3' or 'photons'")

        # Set up input data
        if data is not None:
            dfs = [(data, None)]
            save_results = False
        else:
            # Set up directories
            traced_photons_dir = self.archive / "TracedPhotons"
            saturated_photons_dir = self.archive / "SaturatedPhotons"
            sim_photons_dir = self.archive / "SimPhotons"
            saturated_photons_dir.mkdir(parents=True, exist_ok=True)

            # Find all non-empty traced_data_*.csv files
            csv_files = sorted(traced_photons_dir.glob("traced_sim_data_*.csv"))
            dfs = []
            for f in csv_files:
                if f.stat().st_size > 100:
                    sim_file = sim_photons_dir / f.name.replace("traced_", "")
                    if not sim_file.exists():
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"Warning: Corresponding SimPhotons file {sim_file.name} not found for {f.name}")
                        sim_df = None
                    else:
                        try:
                            sim_df = pd.read_csv(sim_file)
                            if sim_df.empty:
                                sim_df = None
                                if verbosity > VerbosityLevel.QUIET:
                                    print(f"Warning: SimPhotons file {sim_file.name} is empty")
                            else:
                                # Validate nz and pz in SimPhotons
                                missing_cols = [col for col in ['nz', 'pz'] if col not in sim_df.columns]
                                if missing_cols:
                                    if verbosity > VerbosityLevel.QUIET:
                                        print(f"Warning: SimPhotons file {sim_file.name} missing columns {missing_cols}. Setting to NaN.")
                                    for col in missing_cols:
                                        sim_df[col] = np.nan
                        except Exception as e:
                            if verbosity > VerbosityLevel.QUIET:
                                print(f"Error reading SimPhotons file {sim_file.name}: {str(e)}")
                            sim_df = None
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            dfs.append((df, sim_df))
                    except Exception as e:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"Error reading {f.name}: {str(e)}")
                        continue

            if not dfs:
                if verbosity > VerbosityLevel.QUIET:
                    print("No valid traced photon files found in 'TracedPhotons' directory.")
                    print(f"Searched in: {traced_photons_dir}")
                    print("Expected files matching pattern: traced_sim_data_*.csv")
                return None
            save_results = True

        if verbosity > VerbosityLevel.QUIET and save_results:
            print(f"Found {len(dfs)} valid traced photon files to process")

        all_results = []

        # Progress bar for file processing
        file_desc = f"Processing {len(dfs)} files" if save_results else "Processing provided data"
        file_iter = tqdm(dfs, desc=file_desc, disable=not verbosity >= VerbosityLevel.BASIC)

        for i, (df, sim_df) in enumerate(file_iter):
            file_name = f"provided_data_{i}.csv" if not save_results else Path(df.name).name
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nProcessing: {file_name}")

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
                if verbosity > VerbosityLevel.QUIET:
                    print(f"Skipping {file_name}: missing columns {missing_cols}")
                continue

            # Remove rows with NaN in required columns
            df = df.dropna(subset=required_cols)

            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {file_name}: no valid data after removing NaNs")
                continue

            # Check if data is sorted by toa2
            is_sorted = (df['toa2'].diff().dropna() >= 0).all()
            if not is_sorted:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Data in {file_name} is not sorted by toa2. Sorting now.")
                df = df.sort_values('toa2').reset_index(drop=True)

            # Map nz and pz if sim_df is available or if columns exist in df
            neutron_id_to_nz_pz = {}
            if sim_df is not None and 'neutron_id' in sim_df.columns:
                # Validate consistency of nz and pz per neutron_id
                grouped = sim_df.groupby('neutron_id')
                for neutron_id, group in grouped:
                    if 'nz' in group.columns:
                        unique_nz = group['nz'].nunique(dropna=True)
                        if unique_nz > 1:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Warning: Inconsistent nz values for neutron_id {neutron_id} in SimPhotons file")
                        neutron_id_to_nz_pz[neutron_id] = {
                            'nz': group['nz'].iloc[0] if 'nz' in group.columns else np.nan,
                            'pz': group['pz'].iloc[0] if 'pz' in group.columns else np.nan
                        }
            elif 'neutron_id' in df.columns and all(col in df.columns for col in ['nz', 'pz']):
                # Use nz and pz from input DataFrame if available
                grouped = df.groupby('neutron_id')
                for neutron_id, group in grouped:
                    unique_nz = group['nz'].nunique(dropna=True)
                    if unique_nz > 1:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Warning: Inconsistent nz values for neutron_id {neutron_id} in input data")
                    neutron_id_to_nz_pz[neutron_id] = {
                        'nz': group['nz'].iloc[0],
                        'pz': group['pz'].iloc[0]
                    }
            else:
                if verbosity >= VerbosityLevel.QUIET:
                    print(f"Warning: No nz/pz data available for {file_name}. Setting to NaN.")
                neutron_id_to_nz_pz = {nid: {'nz': np.nan, 'pz': np.nan} for nid in df['neutron_id'].unique()}

            # Determine pixel size if not provided
            if pixel_size is None:
                x_range = df['x2'].max() - df['x2'].min()
                y_range = df['y2'].max() - df['y2'].min()
                pixel_size = max(x_range, y_range) / 256.0 if x_range > 0 and y_range > 0 else 1.0

            # Assign pixels (0-based indexing) using vectorized operations
            df['pixel_x'] = np.clip((df['x2'] / pixel_size + 128).astype(int), 0, 255)
            df['pixel_y'] = np.clip((df['y2'] / pixel_size + 128).astype(int), 0, 255)

            # Group photons by pixel and deadtime
            grouped_data = []
            current_group_id = 0

            # Group by pixel_x and pixel_y
            grouped_pixels = df.groupby(['pixel_x', 'pixel_y'])
            
            for (pixel_x, pixel_y), pixel_df in grouped_pixels:
                group_indices = []
                current_time = None

                for idx, row in pixel_df.iterrows():
                    if current_time is None:
                        current_time = row['toa2']
                        group_indices = [idx]
                    elif row['toa2'] <= current_time + deadtime:
                        group_indices.append(idx)
                    else:
                        if group_indices:
                            grouped_data.append((pixel_x, pixel_y, group_indices))
                            for g_idx in group_indices:
                                df.loc[g_idx, 'group_id'] = current_group_id
                            current_group_id += 1
                        current_time = row['toa2']
                        group_indices = [idx]

                if group_indices:
                    grouped_data.append((pixel_x, pixel_y, group_indices))
                    for g_idx in group_indices:
                        df.loc[g_idx, 'group_id'] = current_group_id
                    current_group_id += 1

            if not grouped_data:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"No valid photon groups found in {file_name}")
                continue

            # Process groups based on output format
            result_rows = []

            if output_format == "tpx3":
                for pixel_x, pixel_y, indices in grouped_data:
                    group_df = df.loc[indices]
                    if group_df.empty:
                        continue

                    first_toa = group_df['toa2'].min()
                    last_toa = group_df['toa2'].max()
                    toa_bin = int(first_toa / 1.5625)
                    tot = max(last_toa - first_toa, min_tot)

                    result_rows.append({
                        'x': pixel_x + 1,
                        'y': pixel_y + 1,
                        'toa': toa_bin,
                        'tot': tot
                    })

            elif output_format == "photons":
                for pixel_x, pixel_y, indices in grouped_data:
                    group_df = df.loc[indices]
                    if group_df.empty:
                        continue

                    first_toa = group_df['toa2'].min()
                    last_toa = group_df['toa2'].max()
                    time_diff = last_toa - first_toa

                    if time_diff > deadtime + 1e-6:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Warning: time_diff {time_diff:.2f} ns exceeds deadtime {deadtime} ns "
                                  f"in pixel ({pixel_x}, {pixel_y}) of {file_name}")

                    first_row = group_df.iloc[0]
                    mean_x2 = group_df['x2'].mean()
                    mean_y2 = group_df['y2'].mean()
                    mean_z2 = group_df['z2'].mean()
                    photon_count = len(group_df)
                    neutron_id = first_row['neutron_id']

                    # Get nz and pz for this neutron_id
                    nz_pz = neutron_id_to_nz_pz.get(neutron_id, {'nz': np.nan, 'pz': np.nan})

                    result_rows.append({
                        'x2': mean_x2,
                        'y2': mean_y2,
                        'z2': mean_z2,
                        'id': first_row['id'],
                        'neutron_id': neutron_id,
                        'toa2': first_row['toa2'],
                        'photon_count': photon_count,
                        'time_diff': time_diff,
                        'nz': nz_pz['nz'],
                        'pz': nz_pz['pz']
                    })

            # Create result DataFrame
            result_df = pd.DataFrame(result_rows)

            # Save results to file if processing files
            if save_results:
                output_file = saturated_photons_dir / f"saturated_{file_name}"
                result_df.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Saved results to: {output_file}")

            all_results.append(result_df)

        # Return combined results if requested
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            if verbosity > VerbosityLevel.QUIET:
                print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
            return combined_df

        if verbosity > VerbosityLevel.QUIET:
            print(f"\nProcessing complete. Results saved to: {saturated_photons_dir}")
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
            if verbose > VerbosityLevel.BASIC:
                print(f"Z-scan completed. Best {scan_type}: {best_focus:.2f} with std: {min_std:.3f}")
        else:
            if verbose > VerbosityLevel.BASIC:
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


    def plot(self, opm: "OpticalModel" = None, kind: str = "layout",
                                scale: float = None, 
                                is_dark: bool = False, **kwargs) -> None:
        """
        Plot the lens layout or aberration diagrams.

        Args:
            opm (OpticalModel, optional): Optical model to plot. Defaults to self.opm0.
            kind (str): Type of plot ('layout', 'ray', 'opd', 'spot'). Defaults to 'layout'.
            scale (float):  Scale factor for the plot. If None, uses Fit.User_Scale or Fit.All_Same.
            is_dark (bool): Use dark theme for plots. Defaults to False.
            **kwargs: Additional keyword arguments for the figure.
                - dpi (int, optional): Figure resolution. Defaults to 120.
                - figsize (tuple, optional): Figure size as (width, height). Defaults to (8, 2) for layout, (8, 4) for others.
                - frameon (bool, optional): Whether to draw the frame (for layout only). Defaults to False.
                - Other keyword arguments are passed to the plot function.

        Returns:
            None

        Raises:
            ValueError: If opm is None or kind is unsupported.
        """
        opm = opm if opm is not None else self.opm0
        if opm is None:
            raise ValueError("No optical model available to plot (self.opm0 is None).")

        # Set default figsize based on plot kind
        figsize = kwargs.pop("figsize", (8, 2) if kind == "layout" else (8, 4))
        dpi = kwargs.pop("dpi", 120)
        frameon = kwargs.pop("frameon", False)
        # scale = kwargs.pop("scale", 10)
        scale_type = Fit.User_Scale if scale else Fit.All_Same

        # Ensure model is updated and vignetting is applied
        # opm.seq_model.do_apertures = False
        # opm.update_model()
        # apply_paraxial_vignetting(opm)

        if kind == "layout":
            plt.figure(
                FigureClass=InteractiveLayout,
                opt_model=opm,
                frameon=frameon,
                dpi=dpi,
                figsize=figsize,
                do_draw_rays=True,
                do_paraxial_layout=False
            ).plot(**kwargs)
        elif kind == "ray":
            plt.figure(
                FigureClass=RayFanFigure,
                opt_model=opm,
                data_type="Ray",
                scale_type=scale_type,
                is_dark=is_dark,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        elif kind == "opd":
            plt.figure(
                FigureClass=RayFanFigure,
                opt_model=opm,
                data_type="OPD",
                scale_type=scale_type,
                is_dark=is_dark,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        elif kind == "spot":
            # Remove manual ray tracing - let SpotDiagramFigure handle it
            plt.figure(
                FigureClass=SpotDiagramFigure,
                opt_model=opm,
                scale_type=scale_type,
                user_scale_value=scale,
                is_dark=is_dark,
                frameon=frameon,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        else:
            raise ValueError(f"Unsupported plot kind: {kind}, supported kinds are ['layout', 'ray', 'opd', 'spot']")