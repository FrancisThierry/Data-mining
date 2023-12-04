# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Chargement des données à partir d'un fichier CSV
df = pd.read_csv('data/iris.csv')

# # Création d'un DataFrame pandas
# df = pd.DataFrame(data)

# Encodage de la variable cible (variety)
label_encoder = LabelEncoder()
df['variety'] = label_encoder.fit_transform(df['variety'])

# Séparation des données en variables prédictives et cible
X = df.drop('variety', axis=1)
y = df['variety']

# Création du modèle de classification des k plus proches voisins (KNeighborsClassifier)
knn = KNeighborsClassifier(n_neighbors=3)

# Entraînement du modèle
knn.fit(X, y)

# Prédiction sur les mêmes données pour visualiser la classification
predictions = knn.predict(X)

# Calcul de la précision du modèle
accuracy = accuracy_score(y, predictions)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Création d'un nuage de points pour visualiser la classification
plt.scatter(df['sepal.length'], df['sepal.width'], c=predictions, cmap='viridis')
plt.xlabel('sepal.length')
plt.ylabel('sepal.width')
plt.title('Classification des Iris')
plt.show()