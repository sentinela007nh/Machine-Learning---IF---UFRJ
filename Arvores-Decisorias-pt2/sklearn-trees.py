from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot

# Make a decision tree and train it on the iris datasets

# Load the iris dataset and split it into features and target
# Visualize graphically the dataset

iris = load_iris()
ax = matplotlib.pyplot.subplot()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
matplotlib.pyplot.show()


X, y = iris.data, iris.target
