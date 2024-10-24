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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import cv2
from numpy.polynomial.legendre import Legendre

def data_load(dcm_folder):
    dicoms_data = []
    for root, dirs, files in os.walk(dcm_folder):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    pixel_array = ds.pixel_array
                    if pixel_array.ndim == 3:  # 用于fMRI-BOLD-scan格式数据
                        dicoms_data.append(pixel_array)
                    if pixel_array.ndim == 2:  # 用于SV2A-study和THC_Challenge格式的数据
                        data_array = pixel_array
                        num_blocks_per_side = 7
                        block_size = data_array.shape[0] // num_blocks_per_side
                        pixel_array = []
                        for i in range(num_blocks_per_side):
                            for j in range(num_blocks_per_side):
                                block = data_array[i * block_size:(i + 1) * block_size,
                                        j * block_size:(j + 1) * block_size]
                                pixel_array.append(block)
                        dicoms_data.append(pixel_array)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    dicoms_data = np.array(dicoms_data)
    return dicoms_data


def mask_self_defined(subject_data, threshold_scale=1.3):
    max_value = np.max(subject_data)
    mean_slices = np.mean(subject_data, axis=0)
    thresholds = threshold_scale * np.mean(mean_slices, axis=(1, 2), keepdims=True)
    mask_3d = np.where(mean_slices < thresholds, 0, mean_slices / max_value)
    # mask_3d = np.where(mean_slices < thresholds, 0, 1)

    # binary_mask_array = (mean_slices > thresholds).astype(int)
    # mask_3d = label(binary_mask_array)

    plot_mask_flag = False
    if plot_mask_flag == True:
        fig, axs = plt.subplots(6, 7, figsize=(8, 8))
        for i, ax in enumerate(axs.flat):
            ax.imshow(mask_3d[i], cmap='gray', vmin=0, vmax=1)
            # ax.set_title(f'Slice {i + 1}', fontsize=10)
            ax.axis('off')
        plt.subplots_adjust(hspace=-0.5, wspace=0.)
        plt.savefig('subject_mask.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    return mask_3d


def detrending_signal(subject_data):
    time = subject_data.shape[0]
    slice_num = subject_data.shape[1]
    row = subject_data.shape[2]
    column = subject_data.shape[3]
    detrended_signal = np.zeros((time,slice_num,row,column))
    for i in range(slice_num):
        for j in range(row):
            for h in range(column):
                p = Legendre.basis(2).fit(range(time), subject_data[:,i,j,h], 2)
                trend = p(range(time))
                detrended_signal[:,i,j,h] = subject_data[:,i,j,h] - trend
    return detrended_signal

def calculate_voxelwise_sfnr_res3D(subject_data):
    mean_signal = np.mean(subject_data, axis=0)
    # detrended_data = signal.detrend(subject_data, axis=0)
    detrended_data = detrending_signal(subject_data)
    std_signal = np.std(detrended_data, axis=0)
    sfnr = np.where(std_signal != 0, mean_signal / std_signal, 0)
    return sfnr


def calculate_voxelwise_snr_res3D(subject_data):
    mean_signal = np.mean(subject_data, axis=0)
    std_signal = np.std(subject_data, axis=0)
    snr = np.where(std_signal != 0, mean_signal / std_signal, 0)
    return snr


def calculate_fwhm_2d(image):
    max_value = np.max(image)
    half_max = max_value / 2
    mask = image >= half_max
    coords = np.argwhere(mask)
    if len(coords) >= 2:
        min_row, min_col = np.min(coords, axis=0)
        max_row, max_col = np.max(coords, axis=0)
        fwhm = np.sqrt((max_row - min_row) ** 2 + (max_col - min_col) ** 2)
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


def calculate_fwhm_res3D(subject, scale=1.8):
    fwhm_results = np.zeros((subject.shape[1], subject.shape[2], subject.shape[3]))
    for i in range(subject.shape[1]):
        for j in range(subject.shape[2]):
            for k in range(subject.shape[3]):
                max_value = np.max(subject[:, i, j, k])
                half_max = scale * max_value / 2
                indices = np.where(subject[:, i, j, k] >= half_max)[0]
                if len(indices) < 2:
                    indices = [0]
                fwhm = indices[-1] - indices[0]
                fwhm_results[i, j, k] = fwhm
    fwhm_results = np.max(fwhm_results) - fwhm_results
    return fwhm_results


def calculate_perAF_res3D(subject):
    mean_signal = np.mean(subject, axis=0)
    diff = abs(subject - mean_signal)
    res = np.where(mean_signal != 0, diff / mean_signal, 0) * 100
    perAF = np.mean(res, axis=0)
    return perAF

def correlate_fft(x, y):
    # 基于傅立叶变换计算两个信号的相关性系数
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    fxy = fx * np.conj(fy)
    corr = np.fft.ifft(fxy)
    corr = np.real(corr) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
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
                correlation = np.corrcoef(voxel1, voxel2)
                correlation[np.isnan(correlation)] = 0
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
                min_distance = 1
            radius_of_decorrelation_slice.append(min_distance)
        radius_of_decorrelation.append(np.nanmean(radius_of_decorrelation_slice))
        '''

        # 找到decorrelation半径
        radius_of_decorrelation.append(np.interp(corr_threshold, correlations[:, 2], correlations[:, 3]))
    average_radius_of_decorrelation = np.mean(radius_of_decorrelation)
    return average_radius_of_decorrelation


def autocorrelation(signal):
    """计算自相关函数"""
    n = len(signal)
    mean = np.mean(signal)
    c0 = np.sum((signal - mean) ** 2) / n
    autocorr = np.correlate(signal - mean, signal - mean, mode='full')[-n:]
    return autocorr / (c0 * np.arange(n, 0, -1))


def calculate_rdc_res3D(subject, threshold=0.5):
    rdc_results = np.zeros((subject.shape[1], subject.shape[2], subject.shape[3]))
    for i in range(subject.shape[1]):
        for j in range(subject.shape[2]):
            for k in range(subject.shape[3]):
                """计算去相关半径"""
                acf = autocorrelation(subject[:, i, j, k])
                rdc = np.where(acf < threshold)[0]
                if len(rdc) > 0:
                    rdc_results[i, j, k] = rdc[0]
                else:
                    rdc_results[i, j, k] = 0
    return rdc_results

# New QA metrics

def calculate_uniformity(image):
    # Uniformity computed over the entire volume
    return np.mean(image) / np.std(image, ddof=1) if np.std(image, ddof=1) != 0 else 0



def calculate_ghosting_v0(image):
    """
    Compute ghosting as a function of variation in edge regions across all slices.
    Ghosting is measured by comparing the mean signal in the outer edge to the mean signal in the central region.
    """
    # Determine the width of the edge region; use a fixed number of pixels if image shape varies
    edge_width = min(image.shape[1] // 10, 5)  # Use at most 5 pixels or 10% of the image width, whichever is smaller

    # Ensure edge width does not exceed half the image dimension to avoid overlapping or exceeding bounds
    edge_width = min(edge_width, image.shape[1] // 2)

    # Extract edge regions (top and bottom slices of each image slice)
    top_edge = image[:, :edge_width, :]
    bottom_edge = image[:, -edge_width:, :]

    # Compute mean signals in the edge regions and the central region
    edge_mean = np.mean(np.concatenate([top_edge, bottom_edge], axis=1))
    central_region = image[:, edge_width:image.shape[1] - edge_width, :]
    central_mean = np.mean(central_region)

    # Avoid division by zero
    if central_mean == 0:
        return 0
    else:
        return edge_mean / central_mean if central_mean != 0 else 0

def calculate_ghosting_2d(image):
    data = image.astype(np.uint8)
    # 1. 边缘检测
    edges = cv2.Canny(data, 100, 200)
    # 2. 分析伪影区域
    # 使用轮廓检测来识别边缘区域
    if cv2.__version__.startswith('4.'):
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 3. 计算鬼影指标
    ghosting_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > data.mean():
            ghosting_area += area
    return ghosting_area


def calculate_ghosting(image_seq):
    total_ghosting_score = 0
    total_images = 0
    for i in range(image_seq.shape[0]):
        for j in range(image_seq.shape[1]):
            image = image_seq[i, j]
            ghosting_area = calculate_ghosting_2d(image)
            total_ghosting_score += ghosting_area
            total_images += 1
    average_ghosting_score = total_ghosting_score / total_images
    return average_ghosting_score

def calculate_entropy(image):
    # Entropy calculated over the entire volume
    histogram, _ = np.histogram(image, bins=256, range=[0, np.max(image)])
    histogram_normalized = histogram / histogram.sum()
    return -np.sum(histogram_normalized * np.log2(histogram_normalized + np.finfo(float).eps))


def calculate_sharpness(image):
    """
    Compute the sharpness of an image by averaging the gradient magnitude across all dimensions.
    The function assumes image to be a 3D array (or similar), where gradients will be calculated for each axis.
    """
    gradients = np.gradient(image)  # This will return a list of arrays, one per dimension
    magnitude = np.sqrt(sum(np.square(g) for g in gradients))  # Compute the magnitude of the gradient vector
    return np.mean(magnitude)  # Return the average of gradient magnitudes across the entire image volume


def calculate_contrast(image):
    # Contrast calculated over the entire volume
    return np.std(image)


def calculate_signal_homogeneity(image):
    # Homogeneity calculated as the inverse of average local variance across all slices
    local_mean = uniform_filter(image, size=3)
    local_sqr_mean = uniform_filter(image ** 2, size=3)
    local_variance = local_sqr_mean - local_mean ** 2
    return 1 / np.mean(local_variance)


def QA_metrics_for_single_subject(subject_data: np.ndarray) -> dict:
    mask_3d = mask_self_defined(subject_data, threshold_scale=1.5)

    for idx in range(subject_data.shape[0]):
        subject_data[idx][mask_3d == 0] = 0

    # calculate QA metrics for each subject
    snr = np.mean(calculate_voxelwise_snr_res3D(subject_data))
    sfnr = np.mean(calculate_voxelwise_sfnr_res3D(subject_data))

    fwhm_space = calculate_fwhm_4d(subject_data)
    fwhm_time = np.mean(calculate_fwhm_res3D(subject_data))
    perAF = np.mean(calculate_perAF_res3D(subject_data))
    rdc_space = calculate_rdc(subject_data)
    rdc_time = np.mean(calculate_rdc_res3D(subject_data))
    # rdc = 0

    # Calculating new QA metrics over the masked 3D volume
    uniformity = calculate_uniformity(subject_data)
    ghosting = calculate_ghosting(subject_data)
    entropy = calculate_entropy(subject_data)
    sharpness = calculate_sharpness(subject_data)
    contrast = calculate_contrast(subject_data)
    signal_homogeneity = calculate_signal_homogeneity(subject_data)

    QA_metrics_dict = {"snr": snr,
                       "sfnr": sfnr,
                       "fwhm_space": fwhm_space,
                       "fwhm_time":fwhm_time,
                       "rdc_space": rdc_space,
                       "rdc_time":rdc_time,
                       "perAF": perAF,
                       "uniformity": uniformity,
                       "ghosting": ghosting,
                       "entropy": entropy,
                       "sharpness": sharpness,
                       "contrast": contrast,
                       "signal_homogeneity": signal_homogeneity
                       }
    return QA_metrics_dict


def QA_metrics_for_single_subject_res3D(subject_data: np.ndarray) -> dict:
    mask_3d = mask_self_defined(subject_data, threshold_scale=1.5)

    for idx in range(subject_data.shape[0]):
        subject_data[idx][mask_3d == 0] = 0

    # calculate QA metrics for each subject
    snr_3d = calculate_voxelwise_snr_res3D(subject_data)
    sfnr_3d = calculate_voxelwise_sfnr_res3D(subject_data)
    fwhm_3d = calculate_fwhm_res3D(subject_data)
    perAF_3d = calculate_perAF_res3D(subject_data)
    rdc_3d = calculate_rdc_res3D(subject_data)

    QA_metrics_dict_res3D = {"snr": snr_3d,
                             "sfnr": sfnr_3d,
                             "fwhm": fwhm_3d,
                             "rdc": rdc_3d,
                             "perAF": perAF_3d
                             }
    return QA_metrics_dict_res3D


def QA_metrics_for_nilearn_data():
    """
    The dataset is an embedded dataset within the Nilearn library, treated as a good dataset.
    """
    output_csv_path = "nilearn_data_QA_metrics.csv"
    subject_num = 25  # Total 155 subjects to be downloaded.
    data = datasets.fetch_development_fmri(n_subjects=subject_num)
    results = []
    for i in range(subject_num):
        print("nilearn subject No.:", i)
        fmri_filenames = data.func[i]
        data_origin = nib.load(fmri_filenames).get_fdata()
        subject_data = data_origin.transpose(3, 1, 2, 0)
        QA_metrics_dict = QA_metrics_for_single_subject(subject_data)
        results.append({
            "File ID": i,
            "SNR": QA_metrics_dict["snr"],
            "SFNR": QA_metrics_dict["sfnr"],
            "FWHM_space": QA_metrics_dict["fwhm_space"],
            "FWHM_time": QA_metrics_dict["fwhm_time"],
            "RDC_space": QA_metrics_dict["rdc_space"],
            "RDC_time": QA_metrics_dict["rdc_time"],
            "perAF": QA_metrics_dict["perAF"],
            "Uniformity": QA_metrics_dict["uniformity"],
            "Ghosting": QA_metrics_dict["ghosting"],
            "Entropy": QA_metrics_dict["entropy"],
            "Sharpness": QA_metrics_dict["sharpness"],
            "Contrast": QA_metrics_dict["contrast"],
            "Signal Homogeneity": QA_metrics_dict["signal_homogeneity"]
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, mode='w', header=True, index=False)


def QA_metrics_for_SV2A_data(root_dir):
    """
        The dataset is the given dataset, treated as a bad dataset.
    """
    output_csv_path = os.path.basename(os.path.normpath(root_dir)) + '_QA_metrics.csv'

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
            "FWHM_space": QA_metrics_dict["fwhm_space"],
            "FWHM_time": QA_metrics_dict["fwhm_time"],
            "RDC_space": QA_metrics_dict["rdc_space"],
            "RDC_time": QA_metrics_dict["rdc_time"],
            "perAF": QA_metrics_dict["perAF"],
            "Uniformity": QA_metrics_dict["uniformity"],
            "Ghosting": QA_metrics_dict["ghosting"],
            "Entropy": QA_metrics_dict["entropy"],
            "Sharpness": QA_metrics_dict["sharpness"],
            "Contrast": QA_metrics_dict["contrast"],
            "Signal Homogeneity": QA_metrics_dict["signal_homogeneity"]
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, mode='w', header=True, index=False)


def fmri_bold_scan():
    folder = './fMRI-BOLD-scan'
    subject_data = data_load(folder)
    # QA_metrics_dict = QA_metrics_for_single_subject_res3D(subject_data)
    QA_metrics_dict = QA_metrics_for_single_subject(subject_data)


if __name__ == '__main__':
    root_dir = 'SV2A-study-part2'
    # root_dir = 'SV2A-study-part2'
    QA_metrics_for_SV2A_data(root_dir)
    # QA_metrics_for_nilearn_data()
    # fmri_bold_scan()