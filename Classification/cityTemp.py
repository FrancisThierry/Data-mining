import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

# Génération de données factices pour 50 villes et températures aléatoires
random.seed(42)  # Pour reproduire les mêmes résultats aléatoires
cities = [f"Ville_{i}" for i in range(1, 51)]
temperatures = [random.uniform(-10, 35) for _ in range(50)]

# Création du DataFrame à partir des données factices
temperature_data = pd.DataFrame({'Ville': cities, 'Temperature (Celsius)': temperatures})

# Sélectionner la colonne de température pour le clustering
X = temperature_data[['Temperature (Celsius)']]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-Means pour le clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Choix arbitraire de 5 clusters
clusters = kmeans.fit_predict(X_scaled)

# Ajout des clusters au DataFrame
temperature_data['Cluster'] = clusters

# Affichage des clusters de température
plt.figure(figsize=(8, 6))
plt.scatter(range(len(temperature_data)), temperature_data['Temperature (Celsius)'], c=temperature_data['Cluster'], cmap='viridis')
plt.title('Clusters de Température dans 50 Villes')
plt.xlabel('Villes')
plt.ylabel('Température (Celsius)')
plt.colorbar(label='Cluster')
# plt.xticks(range(len(temperature_data)), temperature_data['Ville'], rotation=90)
plt.tight_layout()
plt.show()
