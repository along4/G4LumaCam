import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass
import json
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import tifffile
from roifile import ImagejRoi, ROI_TYPE
from lmfit.models import Model
from scipy.special import erfc
from matplotlib import pyplot as plt

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
        
        # Set photon_files_dir only for non-groupby structures
        if not self._is_groupby:
            self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
        else:
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
                    "TDC1": True
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
                    "time_res_s": 1.5625e-9,
                    "time_limit": 640
                },
            },
            "out_of_focus": {
                "pixel2photon": {
                    "dSpace": 2,
                    "dTime": 5e-08,
                    "nPxMin": 2,
                    "nPxMax": 12,
                    "TDC1": True
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
                    "time_res_s": 1.5625e-9,
                    "time_limit": 640
                },
            },
            "hitmap": {
                "pixel2photon": {
                    "dSpace": 0.001,
                    "dTime": 1e-9,
                    "nPxMin": 1,
                    "TDC1": True
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
                    "time_res_s": 1.5625e-9,
                    "time_limit": 640
                },
            }
        }


    def _run_roi_analysis(self, 
                    tiff_path: str, 
                    roi_zip_path: str, 
                    output_dir: str,
                    verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                    pixel_size_um: float = 120.0,
                    detector_pixels: int = 512) -> None:
        """
        Reads a TIFF stack and applies rectangular ROIs from a .zip file, calculates counts
        (sum of pixel values) per slice per ROI, and saves CSV files with columns: stack
        (1-based), counts, err (sqrt(counts)) in the specified output directory.
        
        If ROI name contains "mtf", performs MTF analysis on the summed image within that ROI.

        Args:
            tiff_path: Absolute path to the TIFF stack file.
            roi_zip_path: Absolute path to the roi.zip file.
            output_dir: Absolute path to the output directory for CSV files.
            verbosity: Controls the level of output during processing.
            pixel_size_um: Pixel size in micrometers (default: 120.0).
            detector_pixels: Number of detector pixels for frequency conversion (default: 512).
        """
        tiff_path = Path(tiff_path)
        roi_zip_path = Path(roi_zip_path)
        output_dir = Path(output_dir)

        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
        if not roi_zip_path.exists():
            raise FileNotFoundError(f"ROI zip file not found: {roi_zip_path}")

        # Read TIFF stack
        stack = tifffile.imread(tiff_path)
        if len(stack.shape) != 3:
            raise ValueError(f"Expected 3D TIFF stack (slices, height, width), got shape {stack.shape}")
        slices, height, width = stack.shape

        # Check if TIFF stack has non-zero values
        if np.all(stack == 0):
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Warning: TIFF stack at {tiff_path} contains all zero values")
            return

        # Read ROIs - fromfile returns a list directly, not a context manager
        rois = ImagejRoi.fromfile(roi_zip_path)
        
        # Ensure rois is a list
        if isinstance(rois, ImagejRoi):
            rois = [rois]
        
        rect_rois = [roi for roi in rois if roi.roitype == ROI_TYPE.RECT]
        n_rois = len(rect_rois)
        if n_rois == 0:
            raise ValueError(f"No rectangular ROIs found in {roi_zip_path}")

        if verbosity >= VerbosityLevel.BASIC:
            print(f"Loaded TIFF stack: {slices} slices, {height}x{width}")
            print(f"Processing {n_rois} rectangular ROIs from {roi_zip_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Output directory: {output_dir}")

        # Create MTF output directory at base_dir level
        base_dir = output_dir.parent
        mtf_output_dir = base_dir / "MTF_calculation"
        
        # Compute summed image for MTF analysis (only if needed)
        mtf_rois = [roi for roi in rect_rois if "mtf" in (roi.name or "").lower()]
        if mtf_rois:
            summed_image = np.sum(stack, axis=0).astype(np.float64)
            # Replace inf and nan with nan for proper handling
            summed_image = np.where(np.isinf(summed_image), np.nan, summed_image)
            mtf_output_dir.mkdir(parents=True, exist_ok=True)
            if verbosity >= VerbosityLevel.BASIC:
                print(f"MTF output directory: {mtf_output_dir}")

        # Process each rectangular ROI
        for i, roi in enumerate(tqdm(rect_rois, desc="Processing ROIs", disable=(verbosity == VerbosityLevel.QUIET))):
            roi_name = roi.name if roi.name else f"ROI_{i+1}"

            # Generate mask for rectangular ROI
            # Use integer coordinates directly from the ROI
            left = max(0, int(np.floor(roi.left)))
            top = max(0, int(np.floor(roi.top)))
            right = min(width, int(np.ceil(roi.right)))
            bottom = min(height, int(np.ceil(roi.bottom)))
            
            if right <= left or bottom <= top:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {roi_name}: invalid rectangle (left={left}, right={right}, top={top}, bottom={bottom})")
                continue

            # Create mask - note: numpy uses [row, col] indexing which is [y, x]
            mask = np.zeros((height, width), dtype=bool)
            mask[top:bottom, left:right] = True
            area = np.sum(mask)
            
            if area == 0:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {roi_name}: zero area mask")
                continue

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Processing {roi_name}: bounds=[{left}:{right}, {top}:{bottom}], area={area}")

            # Calculate counts per slice
            results = []
            for z in range(slices):
                # Get slice - ensure it's float to avoid overflow
                slice_img = stack[z].astype(np.float64)
                
                # Apply mask and calculate sum
                roi_pixels = slice_img[mask]
                sum_val = np.sum(roi_pixels)
                
                # Calculate error (sqrt of counts, treating as Poisson statistics)
                err_val = np.sqrt(max(sum_val, 0))
                
                results.append({
                    "stack": z + 1,  # 1-based indexing to match ImageJ
                    "counts": sum_val,
                    "err": err_val
                })

            # Save results to CSV
            df = pd.DataFrame(results)
            csv_path = output_dir / f"{roi_name}.csv"
            df.to_csv(csv_path, index=False)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Saved: {csv_path} (mean counts: {df['counts'].mean():.2f})")

            # Perform MTF analysis if "mtf" is in the ROI name
            if "mtf" in roi_name.lower():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Performing MTF analysis for {roi_name}")
                
                try:
                    self._perform_mtf_analysis(
                        summed_image=summed_image,
                        roi_name=roi_name,
                        left=left,
                        right=right,
                        top=top,
                        bottom=bottom,
                        output_dir=mtf_output_dir,
                        pixel_size_um=pixel_size_um,
                        detector_pixels=detector_pixels,
                        verbosity=verbosity
                    )
                except Exception as e:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Warning: MTF analysis failed for {roi_name}: {e}")

        if verbosity >= VerbosityLevel.BASIC:
            print(f"Completed: {n_rois} ROI spectra saved to {output_dir}")



    def _perform_mtf_analysis(self,
                            summed_image: np.ndarray,
                            roi_name: str,
                            left: int,
                            right: int,
                            top: int,
                            bottom: int,
                            output_dir: Path,
                            pixel_size_um: float,
                            detector_pixels: int,
                            verbosity: VerbosityLevel) -> None:
        """
        Performs MTF (Modulation Transfer Function) analysis on a knife-edge ROI.
        
        Args:
            summed_image: 2D array of the summed stack
            roi_name: Name of the ROI
            left, right, top, bottom: ROI boundaries
            output_dir: Directory to save MTF results
            pixel_size_um: Pixel size in micrometers
            detector_pixels: Number of detector pixels for frequency conversion
            verbosity: Verbosity level
        """
        # Extract ROI from summed image
        roi_image = summed_image[top:bottom, left:right]
        
        # Sum across columns (knife edge at lower part, so we sum horizontally)
        # This gives us the Edge Spread Function (ESF)
        esf = np.nansum(roi_image, axis=1)
        x = np.arange(len(esf))
        
        # Remove any remaining nan/inf values
        valid_mask = np.isfinite(esf)
        if not np.any(valid_mask):
            raise ValueError("No valid data points in ROI after removing nan/inf")
        
        x_clean = x[valid_mask]
        esf_clean = esf[valid_mask]
        
        # Define erfc model for fitting the Edge Spread Function
        def erfc_model(x, center, width, amplitude, offset):
            return amplitude * 0.5 * erfc((x - center) / (np.sqrt(2) * width)) + offset
        
        # Create lmfit Model
        model = Model(erfc_model)
        
        # Initial parameter guesses
        center_guess = len(esf_clean) / 2
        amplitude_guess = np.nanmax(esf_clean) - np.nanmin(esf_clean)
        offset_guess = np.nanmin(esf_clean)
        width_guess = len(esf_clean) * 0.1
        
        params = model.make_params(
            center=center_guess,
            width=width_guess,
            amplitude=amplitude_guess,
            offset=offset_guess
        )
        
        # Set reasonable bounds
        # params['width'].min = 0.1
        # params['amplitude'].min = 0
        
        # Perform the fit
        result = model.fit(esf_clean, params, x=x_clean)
        
        # Calculate LSF by differentiating the fitted ESF
        fitted_esf = result.eval(x=x_clean)
        lsf = -np.gradient(fitted_esf, x_clean)
        
        # Compute MTF from FFT of LSF
        mtf = np.abs(np.fft.fft(lsf))
        
        # CORRECTED: Frequency conversion
        # FFT gives frequencies in cycles per sample
        # dx = 1 pixel (spacing in x_clean)
        # freq in cycles/pixel needs to be converted to lp/mm
        # Conversion: cycles/pixel × (1000 μm/mm) / (pixel_size_um μm/pixel) = lp/mm
        dx_pixels = 1.0  # spacing in x_clean array (pixels)
        frequencies = np.fft.fftfreq(len(x_clean), d=dx_pixels)  # cycles per pixel
        
        positive_freq_idx = len(mtf) // 2
        mtf = mtf[:positive_freq_idx]
        frequencies = frequencies[:positive_freq_idx]
        
        if mtf[0] > 0:
            mtf = mtf / mtf[0]
        
        # Convert from cycles/pixel to lp/mm
        # 1 cycle/pixel = 1000/pixel_size_um lp/mm
        frequencies_lpmm = frequencies * 1000.0 / pixel_size_um
        
        # Additional diagnostic info
        nyquist_lpmm = 0.5 * 1000.0 / pixel_size_um  # Nyquist frequency in lp/mm
        
        fit_results = {
            'parameter': ['center', 'width', 'amplitude', 'offset', 'reduced_chi2', 
                        'pixel_size_um', 'nyquist_lpmm', 'roi_height_pixels'],
            'value': [
                result.params['center'].value,
                result.params['width'].value,
                result.params['amplitude'].value,
                result.params['offset'].value,
                result.redchi,
                pixel_size_um,
                nyquist_lpmm,
                bottom - top
            ],
            'stderr': [
                result.params['center'].stderr if result.params['center'].stderr else np.nan,
                result.params['width'].stderr if result.params['width'].stderr else np.nan,
                result.params['amplitude'].stderr if result.params['amplitude'].stderr else np.nan,
                result.params['offset'].stderr if result.params['offset'].stderr else np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ]
        }
        
        fit_df = pd.DataFrame(fit_results)
        fit_csv_path = output_dir / f"{roi_name}_fit_params.csv"
        fit_df.to_csv(fit_csv_path, index=False)
        
        mtf_data = pd.DataFrame({
            'frequency_lpmm': frequencies_lpmm,
            'mtf': mtf
        })
        mtf_csv_path = output_dir / f"{roi_name}_mtf.csv"
        mtf_data.to_csv(mtf_csv_path, index=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        
        # Plot 1: ESF and fit
        axes[0, 0].plot(x_clean, esf_clean, 'o', label='Data', markersize=3, alpha=0.6)
        axes[0, 0].plot(x_clean, fitted_esf, 'r-', label='Fit', linewidth=2)
        axes[0, 0].set_xlabel('Position (pixels)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].set_title(f'Edge Spread Function - {roi_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        # Add text with width in pixels and physical units
        width_mm = result.params['width'].value * pixel_size_um / 1000.0
        axes[0, 0].text(0.05, 0.95, f"Width: {result.params['width'].value:.1f} px ({width_mm:.3f} mm)", 
                        transform=axes[0, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: LSF
        axes[0, 1].plot(x_clean, lsf, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Position (pixels)')
        axes[0, 1].set_ylabel('Intensity Derivative')
        axes[0, 1].set_title('Line Spread Function')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: MTF - CORRECTED X-AXIS LIMIT
        axes[1, 0].plot(frequencies_lpmm, mtf, 'g-', linewidth=2)
        axes[1, 0].axhline(0.1, color='0.7', linestyle='--', zorder=-1, label='MTF=0.1')
        axes[1, 0].set_xlabel('Frequency (lp/mm)')
        axes[1, 0].set_ylabel('MTF')
        axes[1, 0].set_title('Modulation Transfer Function')
        # Use Nyquist frequency as upper limit, or reasonable fraction
        max_freq = min(nyquist_lpmm, 5.0)  # Show up to 5 lp/mm or Nyquist
        axes[1, 0].set_xlim(0, max_freq)
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fit statistics
        axes[1, 1].axis('off')
        fit_text = f"Fit Parameters:\n"
        fit_text += f"Center: {result.params['center'].value:.2f}"
        if result.params['center'].stderr:
            fit_text += f" ± {result.params['center'].stderr:.2f}"
        fit_text += " px\n"
        
        fit_text += f"Width: {result.params['width'].value:.2f}"
        if result.params['width'].stderr:
            fit_text += f" ± {result.params['width'].stderr:.2f}"
        fit_text += f" px ({width_mm:.3f} mm)\n"
        
        fit_text += f"Amplitude: {result.params['amplitude'].value:.2f}"
        if result.params['amplitude'].stderr:
            fit_text += f" ± {result.params['amplitude'].stderr:.2f}"
        fit_text += "\n"
        
        fit_text += f"Offset: {result.params['offset'].value:.2f}"
        if result.params['offset'].stderr:
            fit_text += f" ± {result.params['offset'].stderr:.2f}"
        fit_text += "\n"
        
        fit_text += f"\nReduced χ²: {result.redchi:.4f}\n"
        fit_text += f"Pixel size: {pixel_size_um:.2f} μm\n"
        fit_text += f"Nyquist: {nyquist_lpmm:.3f} lp/mm"
        
        axes[1, 1].text(0.1, 0.5, fit_text, fontsize=10, verticalalignment='center', 
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = output_dir / f"{roi_name}_mtf_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Saved MTF analysis: {fit_csv_path}")
            print(f"Saved MTF data: {mtf_csv_path}")
            print(f"Saved MTF plot: {plot_path}")
            print(f"Reduced χ²: {result.redchi:.4f}")
            print(f"Edge width: {result.params['width'].value:.2f} pixels ({width_mm:.3f} mm)")
            print(f"Nyquist frequency: {nyquist_lpmm:.3f} lp/mm")

    
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

        output_file = final_dir / "image"
        cmd = [
            str(self.empir_dirpath / "bin/empir_event2image"),
            "-I", str(event_files_dir),
            "-o", str(output_file),
            "--paramsFile", str(params_file)
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
                roifile: Optional[str] = None,
                verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                clean: bool = True,
                **kwargs) -> None:
        """
        Process TPX3 files through the EMPIR pipeline.
        
        If roifile is provided, sets event2image=True and runs _run_roi_analysis on the output TIFF.
        
        Args:
            params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder (ignored for groupby structures)
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            roifile: Optional path to roi.zip file for ROI analysis
            verbosity: Controls output level
            clean: If True, deletes existing processed files
            **kwargs: Additional parameters
        """
        # If roifile is provided, ensure event2image is True
        if roifile is not None:
            event2image = True
            if not Path(roifile).exists():
                raise FileNotFoundError(f"ROI file not found: {roifile}")

        # Auto-detect and handle groupby structure
        if self._is_groupby:
            if verbosity > VerbosityLevel.BASIC:
                print("Detected groupby structure, processing all groups...")
            return self._process_grouped(
                params=params,
                n_threads=n_threads,
                suffix=suffix,
                pixel2photon=pixel2photon,
                photon2event=photon2event,
                event2image=event2image,
                export_photons=export_photons,
                export_events=export_events,
                roifile=roifile,
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
            roifile=roifile,
            verbosity=verbosity,
            clean=clean,
            **kwargs
        )

    def _process_grouped(self, 
                        params: Union[str, Dict[str, Any]] = None,
                        n_threads: int = 1,
                        suffix: str = "",
                        pixel2photon: bool = True,
                        photon2event: bool = True,
                        event2image: bool = False,
                        export_photons: bool = True,
                        export_events: bool = False,
                        roifile: Optional[str] = None,
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
        
        original_archive = self.archive
        original_photon_files_dir = self.photon_files_dir
        
        # Resolve roifile to absolute path
        if roifile:
            roifile = str(Path(roifile).resolve())
        
        group_iter = enumerate(self._groupby_subfolders)
        if verbosity >= VerbosityLevel.BASIC:
            group_iter = enumerate(tqdm(self._groupby_subfolders, 
                                        desc="Processing groups",
                                        position=0,
                                        leave=True))
        
        for i, group_folder in group_iter:
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\n{'─'*60}")
                print(f"Group {i+1}/{len(self._groupby_subfolders)}: {group_folder.name}")
                print(f"{'─'*60}")
            
            self.archive = group_folder
            if suffix:
                self.photon_files_dir = self.archive / suffix.strip("_") / "photonFiles"
            else:
                self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                tpx3_dir = group_folder / "tpx3Files"
                if not tpx3_dir.exists() or not list(tpx3_dir.glob("*.tpx3")):
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  No TPX3 files found in {group_folder.name}, skipping")
                    continue
                
                self._process_single(
                    params=params,
                    n_threads=n_threads,
                    suffix=suffix,
                    pixel2photon=pixel2photon,
                    photon2event=photon2event,
                    event2image=event2image,
                    export_photons=export_photons,
                    export_events=export_events,
                    roifile=roifile,
                    verbosity=VerbosityLevel.QUIET,
                    clean=clean,
                    **kwargs
                )
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"✓ Completed group '{group_folder.name}'")
            
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"\n✗ Error processing group '{group_folder.name}': {e}")
                if verbosity >= VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
            
            finally:
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
                        roifile: Optional[str] = None,
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
            tpx3_dir = base_dir / "tpx3Files"  # Use existing tpx3Files from group
        else:
            process_dir = base_dir
            tpx3_dir = base_dir / "tpx3Files"

        if not tpx3_dir.exists() or not list(tpx3_dir.glob("*.tpx3")):
            raise FileNotFoundError(f"No .tpx3 files found in {tpx3_dir}")

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
        
        if roifile:
            tiff_path = final_dir / "image"
            output_dir = process_dir / "ROI_spectra"
            if not tiff_path.exists():
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Warning: TIFF file not found at {tiff_path}, skipping ROI analysis")
            else:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Running ROI analysis on {tiff_path} with ROI file {roifile}")
                self._run_roi_analysis(tiff_path, roifile, output_dir, verbosity=verbosity)
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")