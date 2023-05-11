from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate the data.
X, y = make_regression(n_samples=100, n_features=50, n_informative=50, n_targets=1, noise=100.0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Write the code below.
# Linear regression
model = ElasticNet(l1_ratio=0.3)
model.fit(train_X, train_y)


# Output the coefficient of determination for `test_X` and `test_y`.
print("ElasticNet regression:{}".format(model.score(test_X, test_y)))

# Ridge regression
model = Ridge()
model.fit(train_X, train_y)


# Output the coefficient of determination for `test_X` and `test_y`.
print("Ridge regression:{}".format(model.score(test_X, test_y)))