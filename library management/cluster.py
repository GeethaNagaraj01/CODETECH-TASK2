import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
books = pd.read_csv('books.csv')
borrowers = pd.read_csv('borrowers.csv')
records = pd.read_csv('records.csv')
data = pd.merge(records, books, on='Book_ID')
data = pd.merge(data, borrowers, on='Borrower_ID')
columns_to_drop = ['Record_ID', 'Book_ID', 'Borrower_ID', 'Date_Borrowed']
data = data.drop(columns=columns_to_drop)

data.fillna(0, inplace=True)

data = pd.get_dummies(data, columns=['Genre', 'Membership_Type', 'Author'], drop_first=True)

scaler = StandardScaler()
numeric_columns = ['Age', 'Publication_Year']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster_KMeans'] = kmeans.fit_predict(data.drop(columns=['Cluster_KMeans'], errors='ignore'))
silhouette = silhouette_score(data.drop(columns=['Cluster_KMeans']), data['Cluster_KMeans'])
print(f"Silhouette Score (K-Means): {silhouette}")
dbscan = DBSCAN(eps=1.5, min_samples=5)  # Adjust parameters if needed
data['Cluster_DBSCAN'] = dbscan.fit_predict(data.drop(columns=['Cluster_DBSCAN'], errors='ignore'))
n_clusters = len(set(data['Cluster_DBSCAN'])) - (1 if -1 in data['Cluster_DBSCAN'] else 0)
print(f"Number of clusters identified by DBSCAN: {n_clusters}")

if n_clusters > 1:
    db_score = davies_bouldin_score(data.drop(columns=['Cluster_DBSCAN']), data['Cluster_DBSCAN'])
    print(f"Davies-Bouldin Index (DBSCAN): {db_score}")
else:
    print("DBSCAN did not identify multiple clusters. Try adjusting 'eps' and 'min_samples' parameters.")

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore'))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Cluster_KMeans'], cmap='viridis', s=30)
plt.title("K-Means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.subplot(1, 2, 2)
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data['Cluster_DBSCAN'], palette='viridis', s=30)
plt.title("DBSCAN Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.show()
data.to_csv('clustered_library_data.csv', index=False)
print("Clustered data saved to 'clustered_library_data.csv'.")

