import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/temp.txt',sep=';')

data.info()
print( np.array(data['Ville']))
print( np.array(data['Janv']))



# Données des villes
data = {
    'Ville': np.array(data['Ville']),
    'Janv': np.array(data['Janv']),
    'Fevr': np.array(data['Fevr']),
    'Mars': np.array(data['Mars']),
    'Avri': np.array(data['Avri']),
    'Mai': np.array(data['Mai']),
    'Juin': np.array(data['Juin']),
    'Juill': np.array(data['Juil']),
    'Aout': np.array(data['Aout']),
    'Sept': np.array(data['Sept']),
    'Octo': np.array(data['Octo']),
    'Nove': np.array(data['Nove']),
    'Dece': np.array(data['Dece']),
}

# Création du DataFrame à partir des données
df = pd.DataFrame(data)

# Sélection des colonnes pour le clustering (mois de janvier à décembre)
columns = ['Janv', 'Fevr', 'Mars', 'Avri', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Octo', 'Nove', 'Dece']
X = df[columns]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'analyse en composantes principales (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Application de l'algorithme K-Means pour le clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Choix arbitraire de 3 clusters
clusters = kmeans.fit_predict(X_scaled)

# Ajout des clusters au DataFrame
df['Cluster'] = clusters

# Création du diagramme en nuage de points
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', edgecolor='k')
plt.title('Clusters des Villes basé sur les données météorologiques')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.colorbar(label='Cluster')
for i, city in enumerate(df['Ville']):
    plt.text(X_pca[i, 0], X_pca[i, 1], city, fontsize=8, ha='right')
plt.tight_layout()
plt.show()
