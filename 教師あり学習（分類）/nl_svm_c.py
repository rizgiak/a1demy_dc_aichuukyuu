import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Generate the data.
X, y = make_gaussian_quantiles(n_samples=1250, n_features=2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Set the range of `C` hyperparameter values (this time 1e-5,1e-4,1e-3,0.01,0.1,1,10,100,1000,10000).
C_list = [10 ** i for i in range(-5, 5)]

# Prepare an empty list for drawing graphs.
train_accuracy = []
test_accuracy = []

# Write the code below.
for C in C_list:
    model = SVC(C=C, random_state=2)
    model.fit(train_X, train_y)
    train_accuracy.append(model.score(train_X, train_y))
    test_accuracy.append(model.score(test_X, test_y))

# Prepare the graphs.
# The `semilogx ()` changes the scale of x to the scale of 10 to the xth power.
plt.semilogx(C_list, train_accuracy, label="accuracy of train_data")
plt.semilogx(C_list, test_accuracy, label="accuracy of test_data")
plt.title("accuracy with changing C")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.show()