from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate the data
X, y = make_regression(n_samples=100, n_features=100, n_informative=60, n_targets=1, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Write the code below.
# Linear regression
model = LinearRegression()
model.fit(train_X, train_y)

# Output the coefficient of determination for `test_X` and `test_y`.
print("Linear regression:{}".format(model.score(test_X, test_y)))

# Lasso regression
model = Lasso()
model.fit(train_X, train_y)

# Output the coefficient of determination for `test_X` and `test_y`.
print("Lasso regression:{}".format(model.score(test_X, test_y)))