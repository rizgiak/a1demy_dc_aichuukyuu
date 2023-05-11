import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate the sample data
X, Y = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

distortions = []
for i in range(1, 11):                # Calculate the number of clusters 1 to 10 at once.
    km = KMeans(n_clusters=i,
                init="k-means++",     # Select cluster center by `k-means ++` method.
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)                         # Run a clustering.
    distortions.append(km.inertia_)   # Run `km.fit` to get `km.inertia_`.

# Graph plot
plt.plot(range(1, 11), distortions, marker="o")
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()