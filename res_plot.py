import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from QAM_with_masks_v2 import data_load, QA_metrics_for_single_subject_res3D
from matplotlib.colors import LinearSegmentedColormap

def box_plot(csv_path):
    data = pd.read_csv(csv_path)
    data_to_plot = data.iloc[:, 1:-5]
    # data_to_plot = data.iloc[:, -5:]
    plt.figure(figsize=(10, 6))
    data_to_plot.boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplot of CSV Data (Excluding First Column)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

def plot_metric_all_slices_singel_subject(QA_metric, slice_num=None, save_file_name=None):
    folder_path = 'res_plot'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if slice_num == None:
        fig, axs = plt.subplots(6, 7, figsize=(8, 8))
        for i, ax in enumerate(axs.flat):
            ax.imshow(QA_metric[i], cmap='viridis')
            ax.set_title(f'Slice {i + 1}', fontsize=10)
            ax.axis('off')
            cax = ax.imshow(QA_metric[i], cmap='viridis')
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        if save_file_name != None:
            plt.savefig(os.path.join(folder_path, save_file_name+ '.pdf'), format='pdf', bbox_inches='tight')
        plt.show()
    else:
        plt.imshow(QA_metric[slice_num], cmap='viridis', interpolation='nearest')
        plt.colorbar()
        if save_file_name != None:
            plt.savefig(os.path.join(folder_path, save_file_name+ '.pdf'), format='pdf', bbox_inches='tight')
        plt.show()

def QA_metric_all_slices_singel_subject_plot(dcm_folder):
    subject_data = data_load(dcm_folder)
    QA_metrics = QA_metrics_for_single_subject_res3D(subject_data)
    for QA_metric_name in QA_metrics.keys():
        print('QA_metric_name:',QA_metric_name)
        save_file_name = 'all_slices_singel_subject_'+QA_metric_name
        plot_metric_all_slices_singel_subject(QA_metrics[QA_metric_name],save_file_name = save_file_name)

def plot_metric_for_slice_compare(good_QA_metric, bad_QA_metric,QA_metric_name):
    QA_metric_name = QA_metric_name.upper()
    folder_path = 'res_plot/'+ QA_metric_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(good_QA_metric.shape[0]):
        colors = ["lightgray","blue", "yellow","orange", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        vmin = good_QA_metric[i].min()
        vmax = good_QA_metric[i].max()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im1 = axes[0].imshow(good_QA_metric[i], cmap=cmap,alpha=0.7, interpolation='nearest',vmin=vmin, vmax=vmax)
        axes[0].set_title('Good '+QA_metric_name+f'in slice {i + 1}', fontsize=10)
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
        cbar1.set_label(QA_metric_name, rotation=270, labelpad=15)

        im2 = axes[1].imshow(bad_QA_metric[i], cmap=cmap,alpha=0.7, interpolation='nearest',vmin=vmin, vmax=vmax)
        axes[1].set_title('Bad '+QA_metric_name+f'in slice {i + 1}')
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
        cbar2.set_label(QA_metric_name, rotation=270, labelpad=15)

        plt.suptitle('Comparison of '+QA_metric_name, fontsize=16)
        plt.tight_layout()
        file_name = QA_metric_name+f'_slice_{i + 1}'+'.pdf'
        plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')
        # plt.show()
def QA_metrics_res_compare_plot(good_subject_dcm_folder, bad_subject_dcm_folder):
    good_subject_data = data_load(good_subject_dcm_folder)
    good_QA_metrics = QA_metrics_for_single_subject_res3D(good_subject_data)
    bad_subject_data = data_load(bad_subject_dcm_folder)
    bad_QA_metrics = QA_metrics_for_single_subject_res3D(bad_subject_data)

    for QA_metric_name in good_QA_metrics.keys():
    # for QA_metric_name in ['snr']:
        print('QA_metric_name:',QA_metric_name)
        plot_metric_for_slice_compare(good_QA_metrics[QA_metric_name], bad_QA_metrics[QA_metric_name],QA_metric_name)



if __name__ == '__main__':

    # csv_path = "SV2A-study-partI_QA_metrics.csv"
    # csv_path = "SV2A-study-part2_QA_metrics.csv"
    # csv_path = "nilearn_data_QA_metrics.csv"
    # box_plot(csv_path)
    #===============================================

    good_subject_dcm_folder = 'SV2A-study-part2/012/28-BOLD_-_N-Back'
    bad_subject_dcm_folder = 'SV2A-study-partI/014/45-BOLD_-_N-Back'
    # good_subject_dcm_folder = 'SV2A-study-part2/019/34-BOLD_-_N-Back'
    # bad_subject_dcm_folder = 'SV2A-study-partI/014/45-BOLD_-_N-Back'
    # QA_metric_all_slices_singel_subject_plot(bad_subject_dcm_folder)
    QA_metrics_res_compare_plot(good_subject_dcm_folder, bad_subject_dcm_folder)





