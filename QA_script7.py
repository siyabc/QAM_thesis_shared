import os
import sys
import numpy as np
import pandas as pd
import csv
import gc
from pydicom import dcmread
from PIL import Image
# from skimage import filters, exposure
from scipy import ndimage

def initialize_csv(file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        headers = [
            "File ID", "Slice Index", "Mean Signal", "Standard Deviation", "SNR",
            "SFNR", "CNR", "Uniformity", "FWHM", "Ghosting", "Entropy", 
            "Sharpness", "Contrast", "PSNR", "Signal Homogeneity"
        ]
        writer.writerow(headers)

def resize_image(image, target_shape):
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D array for image resizing, but got shape {image.shape}. Please check the data.")
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(target_shape[::-1], Image.NEAREST)
    return np.array(resized_image)

def linear_detrend(time_series):
    x = np.arange(time_series.shape[0])
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, time_series, rcond=None)[0]
    trend = m * x + b
    detrended = time_series - trend
    return detrended

def calculate_cnr(signal1, signal2, noise):
    cnr = (np.abs(np.mean(signal1) - np.mean(signal2))) / np.std(noise)
    return cnr

def calculate_uniformity(image):
    mean_intensity = np.mean(image)
    max_intensity = np.max(image)
    min_intensity = np.min(image)
    uniformity = (max_intensity - min_intensity) / mean_intensity
    return uniformity

def calculate_fwhm_2d(image):
    max_value = np.max(image)
    half_max = max_value / 2
    mask = image >= half_max
    coords = np.argwhere(mask)
    if len(coords) >= 2:
        min_row, min_col = np.min(coords, axis=0)
        max_row, max_col = np.max(coords, axis=0)
        fwhm = np.sqrt((max_row - min_row)**2 + (max_col - min_col)**2)
    else:
        fwhm = np.nan
    return fwhm

def calculate_pixelwise_snr(time_series_data):
    mean_signal = np.mean(time_series_data, axis=0)
    std_signal = np.std(time_series_data, axis=0)
    snr = np.where(std_signal == 0, 0, mean_signal / std_signal)
    return snr

def extract_voxel_signals(mask, image):
    voxel_signals = []
    voxel_coords = np.argwhere(mask)
    for coord in voxel_coords:
        signal = image[tuple(coord)]
        voxel_signals.append(signal)
    return voxel_signals

def calculate_ghosting(image, roi_mask):
    non_roi_mask = np.logical_not(roi_mask)
    roi_signal = image[roi_mask]
    non_roi_signal = image[non_roi_mask]
    ghosting_ratio = np.mean(non_roi_signal) / np.mean(roi_signal)
    return ghosting_ratio

def calculate_entropy(image):
    """Calculate image entropy, which measures the randomness in the image."""
    hist, _ = np.histogram(image, bins=256)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))  # Add small epsilon to avoid log(0)
    return entropy

def calculate_sharpness(image):
    """Sharpness measured by the variance of the Laplacian."""
    laplacian_var = ndimage.laplace(image).var()
    return laplacian_var

def calculate_contrast(image):
    """Contrast measured as the range between the max and min pixel intensities."""
    contrast = np.max(image) - np.min(image)
    return contrast

def calculate_psnr(image, reference_image):
    mse = np.mean((image - reference_image) ** 2)
    if mse == 0:
        return np.inf
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_signal_homogeneity(image):
    """Signal homogeneity calculated as the inverse of the standard deviation."""
    homogeneity = 1 / np.std(image) if np.std(image) != 0 else np.nan
    return homogeneity

def convert_dicom_to_slices(dicom_file):
    ds = dcmread(dicom_file)
    slice_data = ds.pixel_array
    return slice_data, slice_data.ndim

def process_dicom_file(dicom_file, file_count, output_csv_path):
    slices, num_dimensions = convert_dicom_to_slices(dicom_file)
    results = []
    
    if num_dimensions == 3:
        for slice_index in range(slices.shape[0]):
            slice_image = slices[slice_index, :, :]
            mean_signal = np.mean(slice_image)
            std_signal = np.std(slice_image)
            snr = mean_signal / std_signal if std_signal != 0 else 0
            detrended_signal = linear_detrend(slice_image.flatten())
            sfnr = mean_signal / np.std(detrended_signal) if np.std(detrended_signal) != 0 else 0
            cnr = calculate_cnr(slice_image, np.zeros_like(slice_image), detrended_signal)
            uniformity = calculate_uniformity(slice_image)
            fwhm = calculate_fwhm_2d(slice_image)

            # Define a basic ROI mask
            roi_mask = slice_image > np.percentile(slice_image, 80)  # Take top 20% brightest pixels as ROI
            ghosting = calculate_ghosting(slice_image, roi_mask)
            
            # Additional metrics
            entropy = calculate_entropy(slice_image)
            sharpness = calculate_sharpness(slice_image)
            contrast = calculate_contrast(slice_image)
            psnr = calculate_psnr(slice_image, np.zeros_like(slice_image))  # Reference image could be modified
            homogeneity = calculate_signal_homogeneity(slice_image)
            
            results.append({
                "File ID": file_count,
                "Slice Index": slice_index,
                "Mean Signal": mean_signal,
                "Standard Deviation": std_signal,
                "SNR": snr,
                "SFNR": sfnr,
                "CNR": cnr,
                "Uniformity": uniformity,
                "FWHM": fwhm,
                "Ghosting": ghosting,
                "Entropy": entropy,
                "Sharpness": sharpness,
                "Contrast": contrast,
                "PSNR": psnr,
                "Signal Homogeneity": homogeneity
            })

    elif num_dimensions == 2:
        slice_image = slices
        mean_signal = np.mean(slice_image)
        std_signal = np.std(slice_image)
        snr = mean_signal / std_signal if std_signal != 0 else 0
        detrended_signal = linear_detrend(slice_image.flatten())
        sfnr = mean_signal / np.std(detrended_signal) if np.std(detrended_signal) != 0 else 0
        cnr = calculate_cnr(slice_image, np.zeros_like(slice_image), detrended_signal)
        uniformity = calculate_uniformity(slice_image)
        fwhm = calculate_fwhm_2d(slice_image)
        
        # Define a basic ROI mask
        roi_mask = slice_image > np.percentile(slice_image, 80)
        ghosting = calculate_ghosting(slice_image, roi_mask)
        
        # Additional metrics
        entropy = calculate_entropy(slice_image)
        sharpness = calculate_sharpness(slice_image)
        contrast = calculate_contrast(slice_image)
        psnr = calculate_psnr(slice_image, np.zeros_like(slice_image))
        homogeneity = calculate_signal_homogeneity(slice_image)
        
        results.append({
            "File ID": file_count,
            "Slice Index": 0,
            "Mean Signal": mean_signal,
            "Standard Deviation": std_signal,
            "SNR": snr,
            "SFNR": sfnr,
            "CNR": cnr,
            "Uniformity": uniformity,
            "FWHM": fwhm,
            "Ghosting": ghosting,
            "Entropy": entropy,
            "Sharpness": sharpness,
            "Contrast": contrast,
            "PSNR": psnr,
            "Signal Homogeneity": homogeneity
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    # Clear variables and free memory
    del slices, slice_image, detrended_signal, results
    gc.collect()  # Force garbage collection

if len(sys.argv) < 2:
    print("Usage: python QA_script.py <path_to_directory_containing_dicom_files>")
    sys.exit(1)

root_directory = sys.argv[1]

if not os.path.isdir(root_directory):
    print(f"Error: The directory {root_directory} does not exist.")
    sys.exit(1)

output_csv_path = "image_metrics.csv"
initialize_csv(output_csv_path)

file_count = 0

for dirpath, _, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(dirpath, filename)
            process_dicom_file(dicom_file, file_count, output_csv_path)
            file_count += 1

print(f"All DICOM files processed.")
print(f"Total DICOM files analyzed: {file_count}")
print(f"Results saved to {output_csv_path}")
