import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


df = pd.read_csv('data/cars.csv', sep=';')

# Sélectionner les variables pertinentes pour l'analyse PCA
selected_features = ['Displacement', 'Horsepower', 'Weight']
X = df[selected_features]

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer PCA
pca = PCA(n_components=2)  # Réduire en 2 composantes principales
X_pca = pca.fit_transform(X_scaled)

# Préparation des données pour la classification
y = df['Origin']  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Entraîner un modèle de classification (par exemple, régression logistique)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Évaluer le modèle
accuracy = clf.score(X_test, y_test)
print(f"Précision du modèle: {accuracy}")


# Affichage de la frontière de décision
def plot_decision_boundary(X, y, classifier, title):
    h = 0.02  # Pas de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(8, 6))
    # plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Affichage des points de test
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Affichage de la frontière de décision pour le modèle de classification entraîné
plot_decision_boundary(X_test, y_test, clf, 'Decision Boundary - Logistic Regression')