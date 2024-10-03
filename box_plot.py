import pandas as pd
import matplotlib.pyplot as plt


def box_plot(csv_path):
    data = pd.read_csv(csv_path)
    data_to_plot = data.iloc[:, 1:]
    plt.figure(figsize=(10, 6))
    data_to_plot.boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplot of CSV Data (Excluding First Column)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # csv_path = "SV2A-study-partI_QA_metrics.csv"
    csv_path = "nilearn_data_QA_metrics.csv"
    box_plot(csv_path)



