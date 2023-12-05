import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Exemple de phrases en français
phrases = [
    "Le soleil brille aujourd'hui.",
    "J'ai besoin d'une carte graphique Radeon.",
    "La Tour Eiffel est un symbole de la ville de Paris.",
    "Bordeaux est une belle ville.",
    "Is fait froid aujourd'hui."
]

# Vectorisation des phrases (utilisation de TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(phrases)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Affichage des résultats
for i, label in enumerate(kmeans.labels_):
    print(f"Phrase {i+1} {phrases[i]} : Cluster {label+1}")

# Visualisation des clusters (si possible)
# Vous pouvez ajouter du code pour afficher les clusters sur un graphique
# en utilisant les coordonnées des centroïdes ou d'autres méthodes.
