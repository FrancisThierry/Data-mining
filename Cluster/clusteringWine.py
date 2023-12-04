import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger les données depuis un fichier CSV
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=';')

# Sélectionner les caractéristiques 'pH' (acidité) et 'free sulfur dioxide'
X = data[['pH', 'free sulfur dioxide']]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-Means pour le clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Choix arbitraire de 5 clusters
clusters = kmeans.fit_predict(X_scaled)

# Visualisation des clusters en fonction de l'acidité et du sulfure
plt.figure(figsize=(8, 6))
plt.scatter(X['free sulfur dioxide'], X['pH'], c=clusters, cmap='viridis', edgecolor='k')
plt.title('Clustering des Vins Acides et Contenant du Sulfure')
plt.xlabel('Free Sulfur Dioxide')
plt.ylabel('pH (Acidité)')
plt.colorbar(label='Cluster')
plt.show()
