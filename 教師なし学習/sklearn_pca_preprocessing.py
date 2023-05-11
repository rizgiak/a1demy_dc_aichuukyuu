import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df_wine = pd.read_csv("wine.csv", header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

# Create an instance for standardization.
sc = StandardScaler()
# Train the transformation model from the training data and apply it to the validation data.
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Create an instance of principal component analysis.
pca = PCA(n_components=2)
# Train the transformation model from the training data and apply it to the validation data.
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#　Create an instance of logistic regression.
lr = LogisticRegression()
# Training the classification model with training data after dimension reduction.
lr.fit(X_train_pca, y_train)

# Display score.
print(lr.score(X_train_pca, y_train))
print(lr.score(X_test_pca, y_test))