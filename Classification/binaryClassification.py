import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import r2_score

# Charger les données depuis le CSV
data = pd.read_csv('data/data.csv')

data.info()
print(data.head(5))
# Séparer les caractéristiques (X) de la variable cible (y)
X = data[['heures_etude', 'test_preliminaire']]
y = data['reussite_exam']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Obtenir les coefficients du modèle
coef = model.coef_[0]
intercept = model.intercept_

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le R2 score
r2 = r2_score(y_test, y_pred)

print(f"R2 score: {r2}")

# Créer un diagramme de dispersion
plt.figure(figsize=(10, 6))

# Afficher les points de ceux qui ont réussi
plt.scatter(X_test[y_test == 1]['test_preliminaire'], X_test[y_test == 1]['heures_etude'], color='green', label='Réussi')

# Afficher les points de ceux qui ont échoué
plt.scatter(X_test[y_test == 0]['test_preliminaire'], X_test[y_test == 0]['heures_etude'], color='red', label='Échoué')

# Ajouter une légende et des titres
plt.legend()
plt.title('Diagramme de Dispersion - Classification Binaire')
plt.xlabel('Note du Test Préliminaire')
plt.ylabel('Heures d\'Étude')

# # Tracer la frontière de décision
# x_min, x_max = X['test_preliminaire'].min() - 1, X['test_preliminaire'].max() + 1
# y_min, y_max = X['heures_etude'].min() - 1, X['heures_etude'].max() + 1

# # Tracer la frontière de décision
# x_min, x_max = X['test_preliminaire'].min() - 1, X['test_preliminaire'].max() + 1
# y_min, y_max = X['heures_etude'].min() - 1, X['heures_etude'].max() + 1

# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.contour(xx, yy, Z, colors='orange', linewidths=4, levels=[0.5])

# Afficher le diagramme
plt.show()
