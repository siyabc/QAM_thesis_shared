# 计算ACF——HWHM： 先求得slices的ACF，在通过拟合（可选高斯拟合，高斯指数混合拟合）每张slices的信号幅值，来计算x,y方向的FWHM

import numpy as np
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.optimize import curve_fit
from numpy.polynomial.legendre import Legendre
from QAM_with_masks_v2 import data_load,mask_self_defined,QA_metrics_for_SV2A_data
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import welch
from scipy.fft import ifft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd


def detrending_space_signal(space_signal):
    p = Legendre.basis(2).fit(range(len(space_signal)), space_signal, 2)
    trend = p(range(len(space_signal)))
    detrended_space_signal = space_signal - trend
    return detrended_space_signal

def calculate_acf(detrended_space_signal):
    # frequencies, psd = welch(detrended_space_signal, len(detrended_space_signal))
    n = len(detrended_space_signal)
    fft_values = np.fft.fft(detrended_space_signal)
    # 计算功率谱密度
    psd = (1 / ( n)) * np.abs(fft_values) ** 2
    psd[1:n // 2] *= 2
    acf = ifft(psd).real
    acf /= acf[0]
    return acf

def acf_fitting_model(r, a, b, c):
    return a * np.exp(-r ** 2 / (2 * b ** 2)) + (1 - a) * np.exp(-r / c)
def gaussian_exponential(x, amp_g, mean_g, std_g, amp_e, decay_e):
    return amp_g * np.exp(-((x - mean_g) ** 2) / (2 * std_g ** 2)) + amp_e * np.exp(-decay_e * x)

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def calculate_fwhm_from_fitting_model(acf, x):
    # 拟合ACF曲线
    fitting_model = gaussian
    popt, _ = curve_fit(fitting_model, x, acf, maxfev=100000)

    # 计算 FWHM
    # a,std_g,c = popt
    # amp_g, mean_g, std_g, amp_e, decay_e = popt
    amp_g, mean_g, std_g = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * std_g

    # 计算拟合函数的最大值
    # y_fit = fitting_model(x, *popt)
    # max_value = np.max(y_fit)
    # half_max = max_value / 2
    # indices = np.where(y_fit >= half_max)[0]
    # if len(indices) >= 2:
    #     fwhm = indices[-1] - indices[0]
    # else:
    #     fwhm = 0
    return fwhm


def calculate_fwhm_for_single_slice(single_slice_img):
    # x 方向
    fwhm_x =[]
    for i in range(single_slice_img.shape[0]):
        detrended_space_signal = detrending_space_signal(single_slice_img[i,:])
        acf = calculate_acf(detrended_space_signal)
        if np.isnan(acf).any() or np.isinf(acf).any():
            continue
        fwhm = calculate_fwhm_from_fitting_model(acf, np.arange(len(acf)))
        fwhm_x.append(fwhm)

    if len(fwhm_x)==0:
        fwhm_x_stats = {
            'fwhm_x_min': single_slice_img.shape[1],
            'fwhm_x_max': single_slice_img.shape[1],
            'fwhm_x_mean': single_slice_img.shape[1]
        }
    else:
        fwhm_x_stats = {
            'fwhm_x_min': np.min(fwhm_x),
            'fwhm_x_max': np.max(fwhm_x),
            'fwhm_x_mean': np.mean(fwhm_x)
        }

    # y 方向
    fwhm_y = []
    for j in range(single_slice_img.shape[1]):
        detrended_space_signal = detrending_space_signal(single_slice_img[:,j])
        acf = calculate_acf(detrended_space_signal)
        if np.isnan(acf).any() or np.isinf(acf).any():
            continue
        fwhm = calculate_fwhm_from_fitting_model(acf, np.arange(len(acf)))
        fwhm_y.append(fwhm)

    if len(fwhm_x)==0:
        fwhm_y_stats = {
            'fwhm_y_min': single_slice_img.shape[0],
            'fwhm_y_max': single_slice_img.shape[0],
            'fwhm_y_mean': single_slice_img.shape[0]
        }
    else:
        fwhm_y_stats = {
            'fwhm_y_min': np.min(fwhm_y),
            'fwhm_y_max': np.max(fwhm_y),
            'fwhm_y_mean': np.mean(fwhm_y)
        }

    return fwhm_x_stats,fwhm_y_stats

def calculate_fwhm_for_single_subject(subject_data):
    time_points, slices, height, width = subject_data.shape

    fwhm_x_min_list, fwhm_x_max_list, fwhm_x_mean_list = [], [], []
    fwhm_y_min_list, fwhm_y_max_list, fwhm_y_mean_list = [], [], []

    # 循环遍历每个时间点和slice
    interval = subject_data.shape[0] // 10 #采样10个时间点
    for t in range(subject_data.shape[0])[::interval]:
        print("---t:",t)
        for s in range(42):
            image = subject_data[t, s, :, :]
            fwhm_x_stats, fwhm_y_stats = calculate_fwhm_for_single_slice(image)
            fwhm_x_min_list.append(fwhm_x_stats['fwhm_x_min'])
            fwhm_x_max_list.append(fwhm_x_stats['fwhm_x_max'])
            fwhm_x_mean_list.append(fwhm_x_stats['fwhm_x_mean'])
            fwhm_y_min_list.append(fwhm_y_stats['fwhm_y_min'])
            fwhm_y_max_list.append(fwhm_y_stats['fwhm_y_max'])
            fwhm_y_mean_list.append(fwhm_y_stats['fwhm_y_mean'])

    fwhm_x_min = np.mean(fwhm_x_min_list)
    fwhm_x_max = np.mean(fwhm_x_max_list)
    fwhm_x_mean = np.mean(fwhm_x_mean_list)
    fwhm_y_min = np.mean(fwhm_y_min_list)
    fwhm_y_max = np.mean(fwhm_y_max_list)
    fwhm_y_mean = np.mean(fwhm_y_mean_list)
    fwhm_dict = {"fwhm_x_min": fwhm_x_min,
               "fwhm_x_max": fwhm_x_max,
               "fwhm_x_mean": fwhm_x_mean,
               "fwhm_y_min": fwhm_y_min,
               "fwhm_y_max": fwhm_y_max,
               "fwhm_y_mean": fwhm_y_mean}
    return fwhm_dict


def ACF_FWHM_for_SV2A_data(root_dir):
    """
        The dataset is the given dataset, treated as a bad dataset.
    """
    output_csv_path = os.path.basename(os.path.normpath(root_dir)) + ('_acf_fwhm_G.csv')

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

        fwhm_dict = calculate_fwhm_for_single_subject(subject_data)
        results.append({
            "File ID": i,
            "FWHM_x_min": fwhm_dict["fwhm_x_min"],
            "FWHM_x_max": fwhm_dict["fwhm_x_max"],
            "FWHM_x_mean": fwhm_dict["fwhm_x_mean"],
            "FWHM_y_min": fwhm_dict["fwhm_y_min"],
            "FWHM_y_max": fwhm_dict["fwhm_y_max"],
            "FWHM_y_mean": fwhm_dict["fwhm_y_mean"],
        })

        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, mode='w', header=True, index=False)


if __name__ == '__main__':

    root_dir = 'SV2A-study-partI'
    ACF_FWHM_for_SV2A_data(root_dir)
