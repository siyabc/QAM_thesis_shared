import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
from skimage.measure import label, regionprops
from nilearn import image, datasets, input_data
from scipy import signal
import matplotlib.pyplot as plt #library to view images
from skimage.measure import label, regionprops
from skimage.filters import rank, gaussian
from scipy import ndimage as ndi
import pydicom
import os
# from skimage.morphology import watershed, disk
from skimage import exposure
import csv
import pandas as pd

def data_load(dcm_folder):
    dicoms_data = []
    for root, dirs, files in os.walk(dcm_folder):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    pixel_array = ds.pixel_array
                    if pixel_array.ndim==3: # 用于fMRI-BOLD-scan格式数据
                        dicoms_data.append(pixel_array)
                    if pixel_array.ndim==2: # 用于SV2A-study和THC_Challenge格式的数据
                        data_array = pixel_array
                        num_blocks_per_side = 7
                        block_size = data_array.shape[0] // num_blocks_per_side
                        pixel_array = []
                        for i in range(num_blocks_per_side):
                            for j in range(num_blocks_per_side):
                                block = data_array[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                                pixel_array.append(block)
                        dicoms_data.append(pixel_array)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    dicoms_data = np.array(dicoms_data)
    return dicoms_data

def mask_self_defined(subject_data, threshold_scale=1.3):
    mean_slices = np.mean(subject_data, axis=0)
    thresholds = threshold_scale * np.mean(mean_slices, axis=(1, 2), keepdims=True)
    binary_mask_array = (mean_slices > thresholds).astype(int)
    mask_3d = label(binary_mask_array)
    return mask_3d

def get_time_series(subject_data,mask_3d):
    indices = np.where(mask_3d > 0)
    roi_signal = subject_data[:, indices[0], indices[1], indices[2]]
    return roi_signal

def calculate_voxelwise_sfnr(time_series):
    mean_signal = np.mean(time_series, axis=0)
    detrended_data = signal.detrend(time_series)
    std_signal = np.std(detrended_data, axis=0)
    sfnr = mean_signal / std_signal
    return sfnr

def calculate_voxelwise_snr(time_series):
    mean_signal = np.mean(time_series, axis=0)
    std_signal = np.std(time_series, axis=0)
    snr =  mean_signal / std_signal
    finite_mask = np.isfinite(snr)
    filtered_snr = snr[finite_mask]
    return filtered_snr

def calculate_fwhm(x, y):
    half_max = max(y) / 2.0
    left_idx = np.where(y > half_max)[0][0]
    right_idx = np.where(y > half_max)[0][-1]
    return x[right_idx] - x[left_idx]

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
        fwhm = 0
    return fwhm

def calculate_fwhm_4d(image_data):
    fwhm_values = []
    for dim1_idx in range(image_data.shape[0]):
        for dim2_idx in range(image_data.shape[1]):
            image = image_data[dim1_idx, dim2_idx, :, :]
            fwhm_2d = calculate_fwhm_2d(image)
            fwhm_values.append(fwhm_2d)
    average_fwhm = np.mean(fwhm_values)
    return average_fwhm

def calculate_perAF(signal):
    mean_signal = np.mean(signal)
    perAF = 100 * np.std(signal) / mean_signal
    return perAF

def calculate_rdc(time_series):
    correlation_matrix = np.corrcoef(time_series)
    threshold = 0.5
    above_threshold = np.where(correlation_matrix > threshold)
    voxel_positions = np.array([(i % 10, i // 10) for i in range(correlation_matrix.shape[0])])
    distances = []
    for i, j in zip(*above_threshold):
        if i != j:
            distance = np.linalg.norm(voxel_positions[i] - voxel_positions[j])
            distances.append(distance)
    if distances:
        decorrelation_radius = np.mean(distances)
        return decorrelation_radius
    else:
        print('No correlations greater than the threshold were found.')

def QA_metrics_for_single_subject(subject_data: np.ndarray)-> dict:
    mask_3d = mask_self_defined(subject_data, threshold_scale=1.8)
    roi_signal = get_time_series(subject_data, mask_3d)

    for idx in range(subject_data.shape[0]):
        subject_data[idx][mask_3d == 0] = 0

    # calculate QA metrics for each subject
    snr = np.mean(calculate_voxelwise_snr(roi_signal))
    sfnr = np.mean(calculate_voxelwise_sfnr(roi_signal))
    fwhm = calculate_fwhm_4d(subject_data)
    perAF = calculate_perAF(roi_signal)
    rdc = calculate_rdc(roi_signal)

    QA_metrics_dict = {"snr":snr,
                       "sfnr":sfnr,
                       "fwhm":fwhm,
                       "rdc":rdc,
                       "perAF":perAF
                       }
    return QA_metrics_dict

def QA_metrics_for_nilearn_data():
    """
    The dataset is an embedded dataset within the Nilearn library, treated as a good dataset.
    """
    output_csv_path = "nilearn_data_QA_metrics.csv"
    subject_num = 10  # Total 155 subjects to be downloaded.
    data = datasets.fetch_development_fmri(n_subjects=subject_num)
    results = []
    for i in range(subject_num):
        print("nilearn subject No.:",i)
        fmri_filenames = data.func[i]
        data_origin = nib.load(fmri_filenames).get_fdata()
        subject_data = data_origin.transpose(3, 1, 2, 0)
        QA_metrics_dict = QA_metrics_for_single_subject(subject_data)
        results.append({
            "File ID": i,
            "SNR": QA_metrics_dict["snr"],
            "SFNR": QA_metrics_dict["sfnr"],
            "FWHM": QA_metrics_dict["fwhm"],
            "RDC": QA_metrics_dict["rdc"],
            "perAF": QA_metrics_dict["perAF"]
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, mode='w', header= True, index=False)



def QA_metrics_for_SV2A_data(root_dir):
    """
        The dataset is the given dataset, treated as a bad dataset.
    """
    output_csv_path = os.path.basename(os.path.normpath(root_dir))+'_QA_metrics.csv'

    second_level_folders = []
    for first_level_folder in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level_folder)
        if os.path.isdir(first_level_path):
            for second_level_folder in os.listdir(first_level_path):
                second_level_path = os.path.join(first_level_path, second_level_folder)
                if os.path.isdir(second_level_path):
                    second_level_folders.append(second_level_path)

    results = []
    for i, folder in enumerate(second_level_folders):
        print("subjuect No.:", i)
        subject_data = data_load(folder)

        QA_metrics_dict = QA_metrics_for_single_subject(subject_data)
        results.append({
            "File ID": i,
            "SNR": QA_metrics_dict["snr"],
            "SFNR": QA_metrics_dict["sfnr"],
            "FWHM": QA_metrics_dict["fwhm"],
            "RDC": QA_metrics_dict["rdc"],
            "perAF": QA_metrics_dict["perAF"]
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, mode='w', header= True, index=False)


if __name__ == '__main__':
    root_dir = 'SV2A-study-partI'
    QA_metrics_for_SV2A_data(root_dir)
    # QA_metrics_for_nilearn_data()