# 1. Import necessary libraries and dataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the wine dataset
data = load_wine()
X = data.data
y = data.target  # Actual labels (for comparison later)

# 2. Scale variables as needed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method for K-Means and Agglomerative Clustering
inertia_kmeans = []
within_cluster_variance_agg = []
k_values = range(1, 10)

# Calculate inertia for K-Means and within-cluster variance for Agglomerative
for k in k_values:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_kmeans.append(kmeans.inertia_)
    
    # Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=k)
    y_agg = agg_clustering.fit_predict(X_scaled)
    
    # Calculate within-cluster variance for Agglomerative Clustering
    variance = 0
    for i in range(k):
        cluster_points = X_scaled[y_agg == i]
        cluster_center = cluster_points.mean(axis=0)
        variance += np.sum((cluster_points - cluster_center) ** 2)
    within_cluster_variance_agg.append(variance)

# Plot the elbow graphs side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Elbow graph for K-Means
ax1.plot(k_values, inertia_kmeans, marker='o')
ax1.set_title('Elbow Method for K-Means')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia (K-Means)')
ax1.legend(['K-Means Inertia'])

# Elbow graph for Agglomerative Clustering
ax2.plot(k_values, within_cluster_variance_agg, marker='o', color='r')
ax2.set_title('Elbow Method for Agglomerative Clustering')
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Within-Cluster Variance (Agglomerative)')
ax2.legend(['Agglomerative Variance'])

plt.show()

# From the elbow plot, we can select the optimal number of clusters (K=3 for both)

# Initial K-Means clustering with K=3
kmeans_initial = KMeans(n_clusters=3, random_state=42)
y_kmeans_unscaled = kmeans_initial.fit_predict(X)
y_kmeans_initial = kmeans_initial.fit_predict(X_scaled)

# Initial DBSCAN clustering
dbscan_initial = DBSCAN(eps=0.5, min_samples=5)
y_dbscan_initial = dbscan_initial.fit_predict(X_scaled)

# 4. Agglomerative Clustering with 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)
y_agg = agg_clustering.fit_predict(X_scaled)

# Reduce dimensions with PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_unscaled_pca = pca.fit_transform(X)

# Visualize Unscaled vs Scaled Actual Clusters
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

scatter1 = ax1.scatter(X_unscaled_pca[:, 0], X_unscaled_pca[:, 1], c=y, cmap='rainbow')
ax1.set_title('Unscaled Actual Labels')
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Labels")
ax1.add_artist(legend1)

scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
ax2.set_title('Scaled Actual Labels')
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')
legend1 = ax2.legend(*scatter2.legend_elements(), title="Labels")
ax2.add_artist(legend1)

plt.show()




# Visualize K-Means clustering results
fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(10, 5))

# Plot for K-Means Clustering
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans_initial, cmap='viridis')
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
ax1.add_artist(legend1)

# Plot for Actual Labels
scatter4 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
ax4.set_title('Actual Labels')
ax4.set_xlabel('PCA Component 1')
ax4.set_ylabel('PCA Component 2')
legend4 = ax4.legend(*scatter4.legend_elements(), title="Labels")
ax4.add_artist(legend4)

plt.show()


# Visualize DBSCAN results
fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(10, 5))

# Plot for DBSCAN Clustering
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan_initial, cmap='plasma')
ax2.set_title('Initial DBSCAN Clustering')
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')
legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
ax2.add_artist(legend2)

# Plot for Actual Labels
scatter4 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
ax4.set_title('Actual Labels')
ax4.set_xlabel('PCA Component 1')
ax4.set_ylabel('PCA Component 2')
legend4 = ax4.legend(*scatter4.legend_elements(), title="Labels")
ax4.add_artist(legend4)

plt.show()

# Plot for Agglomerative Clustering
# scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y_agg, cmap='viridis')
# ax3.set_title('Initial Agglomerative Clustering')
# ax3.set_xlabel('PCA Component 1')
# ax3.set_ylabel('PCA Component 2')
# legend3 = ax3.legend(*scatter3.legend_elements(), title="Clusters")
# ax3.add_artist(legend3)

# plt.show()





# 5. Grid search for DBSCAN parameters
eps_values = np.linspace(4, 4.2, 50)
min_samples_values = range(3, 7)
best_silhouette = -1
best_eps = 0
best_min_samples = 0

# Perform grid search over eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_dbscan = dbscan.fit_predict(X_scaled)
        
        # DBSCAN returns -1 for noise, so check for valid clustering
        if len(set(y_dbscan)) > 1:  # Check if at least 2 clusters
            silhouette = silhouette_score(X_scaled, y_dbscan)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples

# Use best eps and min_samples to fit DBSCAN again
print("Best eps:", best_eps, "\nBest min_samples:", best_min_samples)
dbscan_tuned = DBSCAN(eps=best_eps, min_samples=best_min_samples)
y_dbscan_tuned = dbscan_tuned.fit_predict(X_scaled)

# 6. Visualize tuned clustering results (DBSCAN and Actual Labels)
fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 5))

# Plot for Tuned DBSCAN Clustering
scatter5 = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan_tuned, cmap='plasma')
ax5.set_title(f'Tuned DBSCAN Clustering (eps={round(best_eps,3)}, min_samples={best_min_samples})')
ax5.set_xlabel('PCA Component 1')
ax5.set_ylabel('PCA Component 2')
legend5 = ax5.legend(*scatter5.legend_elements(), title="Clusters")
ax5.add_artist(legend5)

# Plot for Actual Labels
scatter6 = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
ax6.set_title('Actual Labels')
ax6.set_xlabel('PCA Component 1')
ax6.set_ylabel('PCA Component 2')
legend6 = ax6.legend(*scatter6.legend_elements(), title="Labels")
ax6.add_artist(legend6)

plt.show()


# 7. Calculate silhouette scores for initial and tuned K-Means and DBSCAN
# Initial silhouette scores
silhouette_kmeans_initial = silhouette_score(X_scaled, y_kmeans_initial)
# silhouette_dbscan_initial = silhouette_score(X_scaled, y_dbscan_initial)
silhouette_agg = silhouette_score(X_scaled, y_agg)

# Tuned silhouette scores
silhouette_kmeans_tuned = silhouette_score(X_scaled, y_kmeans_initial)
silhouette_dbscan_tuned = silhouette_score(X_scaled, y_dbscan_tuned)

# Calculate clustering accuracy using ARI and AMI
ari_kmeans_initial = adjusted_rand_score(y, y_kmeans_initial)
ari_dbscan_initial = adjusted_rand_score(y, y_dbscan_initial)
ari_dbscan_tuned = adjusted_rand_score(y, y_dbscan_tuned)
ari_agg = adjusted_rand_score(y, y_agg)

ami_kmeans_initial = adjusted_mutual_info_score(y, y_kmeans_initial)
ami_dbscan_initial = adjusted_mutual_info_score(y, y_dbscan_initial)
ami_dbscan_tuned = adjusted_mutual_info_score(y, y_dbscan_tuned)
ami_agg = adjusted_mutual_info_score(y, y_agg)

# 8. Display scores in table format
# Create a DataFrame to hold the metrics
results = {
    "Clustering Method": ["K-Means (Initial)", "K-Means (Tuned)", "DBSCAN (Initial)", "DBSCAN (Tuned)", "Agglomerative"],
    "Silhouette Score": [silhouette_kmeans_initial, silhouette_kmeans_tuned, -1, silhouette_dbscan_tuned, silhouette_agg],
    "ARI (Adjusted Rand Index)": [ari_kmeans_initial, ari_kmeans_initial, ari_dbscan_initial, ari_dbscan_tuned, ari_agg],
    "AMI (Adjusted Mutual Information)": [ami_kmeans_initial, ami_kmeans_initial, ami_dbscan_initial, ami_dbscan_tuned, ami_agg]
}

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Display the table
print(df_results)
