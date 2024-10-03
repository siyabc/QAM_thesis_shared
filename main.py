import os
import sys
from QAM_with_masks import QA_metrics_for_SV2A_data,QA_metrics_for_nilearn_data


if len(sys.argv) < 2:
    print("Usage: python main.py <path_to_directory_containing_dicom_files>")
    sys.exit(1)

root_directory = sys.argv[1]
if not os.path.isdir(root_directory):
    print(f"Error: The directory {root_directory} does not exist.")
    sys.exit(1)
QA_metrics_for_SV2A_data(root_directory)