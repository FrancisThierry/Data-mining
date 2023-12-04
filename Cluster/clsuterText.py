from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors


documents = ["the young french men crowned world champions",
             "Google Translate app is getting more intelligent everyday",
             "Facebook face recognition is driving me crazy",
             "who is going to win the Golden Ball title this year",
             "these camera apps are funny",
             "Croacian team made a brilliant world cup campaign reaching the final match",
             "Google Chrome extensions are useful.",
             "Social Media apps leveraging AI incredibly",
             "Qatar 2022 FIFA world cup is played in winter"]
 
 
vectorizer = TfidfVectorizer(stop_words = 'english')
data = vectorizer.fit_transform(documents)
 
true_k = 2
clustering_model = KMeans(n_clusters = true_k, 
                          init = 'k-means++',
                          max_iter = 300, n_init = 10)
clustering_model.fit(data)

## terms per cluster

sorted_centroids = clustering_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in sorted_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
        print()
 
# Réduction de dimensionnalité à l'aide de PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data.toarray())

# Création du graphique de dispersion
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=clustering_model.labels_)

# Création de tooltips avec mplcursors
labels = [", ".join([terms[ind] for ind in sorted_centroids[cluster, :10]]) for cluster in clustering_model.labels_]
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

plt.title('Clusters with Labels (Hover for Tooltip)')
plt.grid(True)
plt.show()