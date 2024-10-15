import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('nilearn_data_QA_metrics.csv')
df1 = pd.read_csv('SV2A-study-partI_QA_metrics.csv')
df1 = df1.iloc[:, 1:]
df1 = df1.fillna(0)
df1 = df1.replace([np.inf, -np.inf], 0)

df2 = pd.read_csv('SV2A-study-part2_QA_metrics.csv')
df2 = df2.iloc[:, 1:]
df2 = df2.fillna(0)
df2 = df2.replace([np.inf, -np.inf], 0)

df = pd.concat([df1, df2], axis=0, ignore_index=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA()
pca.fit(scaled_data)

# 计算每个主成分的贡献度（方差比）
explained_variance = pca.explained_variance_ratio_
# print("explained_variance:",explained_variance)

# 计算累计贡献度
cumulative_variance = np.cumsum(explained_variance)

# 找到贡献度前85%的主成分
n_components = np.argmax(cumulative_variance >= 0.85) + 1
components = pca.components_[:n_components]
feature_contributions = pd.DataFrame(components, columns=df.columns)

important_features = {}
for i in range(n_components):
    component = feature_contributions.iloc[i]
    print(f"\n主成分 {i + 1} 的特征贡献度:")
    print(feature_contributions.iloc[i])

for i, component in enumerate(components):
    # 获取特征及其贡献度
    feature_contributions = pd.Series(component, index=df.columns)
    top_features = feature_contributions.abs().nlargest(3)
    print(f"主成分 {i + 1} 的贡献度前三个特征：")
    print(top_features)

total_contribution = np.abs(components).sum(axis=0)
total_contribution_normalized = total_contribution / total_contribution.sum()
contribution_df = pd.DataFrame({
    'Feature': df.columns,
    'Total Contribution': total_contribution_normalized
})
print("\n各特征的总贡献度：")
print(contribution_df.sort_values(by='Total Contribution', ascending=False))

