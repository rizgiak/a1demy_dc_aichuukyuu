import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate the sample data.
X, Y = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

# Create a `km` instance from the `KMeans` class.
km = KMeans(n_clusters=10,            # Number of clusters  # Change it. 
            init="random",           # Randomly set the initial value of Centroid_default: "k-means ++"
            n_init=10,               # Number of k-means executions using different centroid initials
            max_iter=300,            # Maximum number of times to repeat the k-means algorithm
            tol=1e-04,               # Relative margin of error for determining convergence
            random_state=0)          # Random number generation initialization

# Run clustering with `fit_predict` method
Y_km = km.fit_predict(X)


# Plot the data according to the cluster number (Y_km).
for n in range(np.max(Y_km)+1):
    plt.scatter(X[Y_km == n, 0], X[Y_km == n, 1], s=50, c=cm.hsv(
        float(n) / 10), marker="*", label="cluster"+str(n+1))

# Plot centroids, `km.cluster_centers_` contains the coordinates of the centroids for each cluster.
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=250, marker="*", c="black", label="centroids")

plt.grid()
plt.show()