# Import the modules.
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate the data.
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Range of values for `n_estimators` (1 to 20)
n_estimators_list = [i for i in range(1, 21)]

# Create an empty list to store the accuracy.
accuracy = []

# Write the code below.
# Train the model while changing `n_estimators`.
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# Plot the graph.
plt.plot(n_estimators_list, accuracy)
plt.title("accuracy by n_estimators increasement")
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.show()