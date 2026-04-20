"""
DICOM file handler for clinical X-ray images.
Supports reading DICOM files, extracting metadata, applying windowing,
and converting to standard image formats for model inference.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings


def is_dicom_file(filepath: str) -> bool:
    """Check if a file is a DICOM file based on extension or magic bytes."""
    path = Path(filepath)
    ext = path.suffix.lower().lstrip('.')
    if ext in ('dcm', 'dicom'):
        return True
    # Check DICOM magic bytes (DICM at offset 128)
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b'DICM'
    except Exception:
        return False


def read_dicom(filepath: str) -> Dict:
    """
    Read a DICOM file and extract image data and metadata.
    
    Args:
        filepath: Path to DICOM file
    
    Returns:
        Dictionary with keys: 'pixel_array', 'metadata', 'image'
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install with: pip install pydicom"
        )
    
    ds = pydicom.dcmread(filepath)
    
    # Extract pixel data
    pixel_array = ds.pixel_array.astype(np.float64)
    
    # Apply rescale slope/intercept if present
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept
    
    # Extract metadata
    metadata = extract_dicom_metadata(ds)
    
    # Convert to displayable image
    image = dicom_to_image(pixel_array, ds)
    
    return {
        'pixel_array': pixel_array,
        'metadata': metadata,
        'image': image,
        'dataset': ds
    }


def extract_dicom_metadata(ds) -> Dict:
    """Extract relevant metadata from DICOM dataset."""
    metadata = {}
    
    fields = {
        'PatientName': 'patient_name',
        'PatientID': 'patient_id',
        'PatientAge': 'patient_age',
        'PatientSex': 'patient_sex',
        'StudyDate': 'study_date',
        'StudyDescription': 'study_description',
        'Modality': 'modality',
        'InstitutionName': 'institution',
        'Manufacturer': 'manufacturer',
        'BodyPartExamined': 'body_part',
        'ViewPosition': 'view_position',
        'Rows': 'rows',
        'Columns': 'columns',
        'BitsAllocated': 'bits_allocated',
        'BitsStored': 'bits_stored',
        'PixelSpacing': 'pixel_spacing',
        'WindowCenter': 'window_center',
        'WindowWidth': 'window_width',
    }
    
    for dicom_field, key in fields.items():
        try:
            value = getattr(ds, dicom_field, None)
            if value is not None:
                # Convert to string or basic type
                if hasattr(value, 'original_string'):
                    metadata[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    metadata[key] = [float(v) for v in value]
                else:
                    metadata[key] = str(value)
        except Exception:
            pass
    
    return metadata


def apply_windowing(pixel_array: np.ndarray, center: float = 40, 
                    width: float = 400) -> np.ndarray:
    """
    Apply windowing (contrast adjustment) to DICOM pixel data.
    
    Args:
        pixel_array: Raw pixel values (HU for CT, raw for X-ray)
        center: Window center (level)
        width: Window width
    
    Returns:
        Windowed array normalized to [0, 255]
    """
    lower = center - width / 2
    upper = center + width / 2
    
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    
    return windowed


def dicom_to_image(pixel_array: np.ndarray, ds=None,
                   window_center: float = None,
                   window_width: float = None) -> Image.Image:
    """
    Convert DICOM pixel array to PIL Image.
    
    Args:
        pixel_array: Raw or windowed pixel array
        ds: DICOM dataset (for extracting windowing params)
        window_center: Manual window center override
        window_width: Manual window width override
    
    Returns:
        PIL Image in RGB mode
    """
    # Try to get windowing from DICOM metadata
    if window_center is None and ds is not None:
        wc = getattr(ds, 'WindowCenter', None)
        if wc is not None:
            window_center = float(wc) if not isinstance(wc, (list, tuple)) else float(wc[0])
    
    if window_width is None and ds is not None:
        ww = getattr(ds, 'WindowWidth', None)
        if ww is not None:
            window_width = float(ww) if not isinstance(ww, (list, tuple)) else float(ww[0])
    
    # Apply windowing if available
    if window_center is not None and window_width is not None:
        img_array = apply_windowing(pixel_array, window_center, window_width)
    else:
        # Simple normalization to 0-255
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            img_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(pixel_array, dtype=np.uint8)
    
    # Handle photometric interpretation (invert if needed)
    if ds is not None:
        pi = getattr(ds, 'PhotometricInterpretation', '')
        if 'MONOCHROME1' in str(pi):
            img_array = 255 - img_array
    
    # Convert grayscale to RGB
    image = Image.fromarray(img_array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def dicom_to_temp_png(dicom_path: str, output_path: str = None) -> str:
    """
    Convert DICOM to temporary PNG file for model inference.
    
    Args:
        dicom_path: Path to DICOM file
        output_path: Output PNG path (auto-generated if None)
    
    Returns:
        Path to the PNG file
    """
    if output_path is None:
        output_path = str(Path(dicom_path).with_suffix('.png'))
    
    result = read_dicom(dicom_path)
    result['image'].save(output_path)
    
    return output_path


def get_dicom_preview_info(filepath: str) -> str:
    """Get a human-readable summary of DICOM metadata."""
    try:
        result = read_dicom(filepath)
        meta = result['metadata']
        
        lines = ["=== DICOM File Info ==="]
        for key, value in meta.items():
            label = key.replace('_', ' ').title()
            lines.append(f"  {label}: {value}")
        
        return '\n'.join(lines)
    except Exception as e:
        return f"Error reading DICOM: {e}"


if __name__ == "__main__":
    print("DICOM handler module loaded successfully")
    print("Features:")
    print("  - Read DICOM files with pydicom")
    print("  - Extract patient/study metadata")
    print("  - Apply windowing (contrast adjustment)")
    print("  - Convert DICOM to PIL Image / PNG")
