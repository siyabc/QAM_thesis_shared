import numpy as np
import nibabel as nib
from nilearn import datasets
from scipy import signal
from skimage.measure import label
import pydicom
import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import zoom
from scipy.stats import pearsonr

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
    non_zero_std_indices = std_signal != 0
    snr = mean_signal[non_zero_std_indices] / std_signal[non_zero_std_indices]
    return snr

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
    mean_signal = np.mean(signal,axis=0)
    diff= signal-mean_signal.reshape(1,len(mean_signal))
    perAF = np.mean(diff/mean_signal)
    return perAF

def correlate_fft(x, y):
    # 基于傅立叶变换计算两个信号的相关性系数
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    fxy = fx * np.conj(fy)
    corr = np.fft.ifft(fxy)
    corr = np.real(corr) / np.sqrt(np.sum(x**2) * np.sum(y**2))
    return corr
def calculate_rdc(data):
    corr_threshold = 0.5
    radius_of_decorrelation = []
    for i in range(data.shape[1]):
        # 获取第i个时间点的3D脑部图像
        image = data[:, i, :, :]
        target_size = 5
        zoom_factor = target_size / data.shape[2]
        downsampled_image = zoom(image, (1, zoom_factor, zoom_factor))
        spatial_distances = squareform(pdist(np.indices((target_size, target_size)).reshape(2, -1).T))
        correlations = []
        # 遍历每对体素
        for j in range(target_size * target_size):
            for k in range(j + 1, target_size * target_size):
                voxel1 = downsampled_image[:, j // target_size, j % target_size]
                voxel2 = downsampled_image[:, k // target_size, k % target_size]
                if np.all(voxel1 == 0) or np.all(voxel2 == 0):
                    correlations.append((j, k, 0, spatial_distances[j, k]))
                    continue
                correlation = np.corrcoef(voxel1, voxel2)
                correlations.append((j, k, abs(correlation[0, 1]), spatial_distances[j, k]))
        correlations = np.array(correlations)
        correlations = correlations[correlations[:, 2].argsort(axis=0)]

        '''
        ## Another method to calculate rdc for each slice
        # 遍历每个体素点
        radius_of_decorrelation_slice = []
        for i in range(downsampled_image.shape[0] * downsampled_image.shape[1]):
            mask = (correlations[:, 0] == i) | (correlations[:, 1] == i)
            voxel_correlations = correlations[mask]
            mask = voxel_correlations[:, 2] < corr_threshold
            if np.any(mask):
                min_distance = np.min(voxel_correlations[mask, 3])
            else:
                min_distance = np.nan
            radius_of_decorrelation_slice.append(min_distance)
        radius_of_decorrelation.append(np.nanmean(radius_of_decorrelation_slice))
        '''

        # 找到decorrelation半径
        radius_of_decorrelation.append(np.interp(corr_threshold, correlations[:, 2], correlations[:, 3]))
    average_radius_of_decorrelation = np.mean(radius_of_decorrelation)
    return average_radius_of_decorrelation


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
    rdc = calculate_rdc(subject_data)
    # rdc = 0

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
    subject_num = 25  # Total 155 subjects to be downloaded.
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
    # root_dir = 'SV2A-study-partI'
    # QA_metrics_for_SV2A_data(root_dir)

    QA_metrics_for_nilearn_data()