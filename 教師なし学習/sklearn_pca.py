import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import PCA.
# - -------------------------
from sklearn.decomposition import PCA
# - -------------------------

df_wine = pd.read_csv("wine.csv", header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Create an instance of principal component analysis. Set the number of principal components to 2.
pca = PCA(n_components=2)

# Train the transformation model from the data and transform it.
X_pca = pca.fit_transform(X)

# Visualization
color = ["r", "b", "g"]
marker = ["s", "x", "o"]
for label, color, marker in zip(np.unique(y), color, marker):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                c=color, marker=marker, label=label)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.show()

print(X_pca)  # Do not delete the code below to check the execution result.