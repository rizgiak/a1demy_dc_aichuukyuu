import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Generate monthly data.
X, Y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Define graph and two axes. The `ax1` on the left is for k-means method, the `ax2` on the right is for DBSCAN.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# k-means method
km = KMeans(n_clusters=2, random_state=0)
Y_km = km.fit_predict(X)

ax1.scatter(X[Y_km == 0, 0], X[Y_km == 0, 1], c="lightblue",
            marker="o", s=40, label="cluster 1")
ax1.scatter(X[Y_km == 1, 0], X[Y_km == 1, 1], c="red",
            marker="s", s=40, label="cluster 2")
ax1.set_title("K-means clustering")
ax1.legend()

# Run a clustering with DBSCAN.  # Complete the code.
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric="euclidean")
Y_db = db.fit_predict(X)

ax2.scatter(X[Y_db == 0, 0], X[Y_db == 0, 1], c="lightblue",
            marker="o", s=40, label="cluster 1")
ax2.scatter(X[Y_db == 1, 0], X[Y_db == 1, 1], c="red",
            marker="s", s=40, label="cluster 2")
ax2.set_title("DBSCAN clustering")
ax2.legend()
plt.show()