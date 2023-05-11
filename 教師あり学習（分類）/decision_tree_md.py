# Import the modules.
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate the data.
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Range of values for `max_depth` (1 to 10)
depth_list = [i for i in range(1, 11)]

# Create an empty list to store the accuracy.
accuracy = []

# Write the code below.
# Train the model while changing `max_depth`.
for max_depth in depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# That's all for editing the code.
# Plot the graph.
plt.plot(depth_list, accuracy)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("accuracy by changing max_depth")
plt.show()