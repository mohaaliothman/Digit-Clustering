#score=0.55
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

data = pd.read_csv('/kaggle/input/digit-clustering/data.csv')
id=data['ID']
X = data.drop(columns=['ID'])

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
pca = PCA(n_components=0.945)  
X_pca = pca.fit_transform(X_normalized)

hierarchical = AgglomerativeClustering(distance_threshold=38,n_clusters=None)
clustersh = hierarchical.fit_predict(X_pca)
Counter(hierarchical.labels_)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clustersh, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA-Transformed Data with Hierarchical Clusters')
plt.colorbar(label='Cluster Label')
plt.show()

output_df = pd.DataFrame({'Label': clustersh}, index=id)

output_df.to_csv('cluster_predictions_hierarchical.csv', index=True)
output_df
