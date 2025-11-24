import sklearn
import matplotlib.pyplot
import graphviz
import io

# Make a decision tree and train it on the iris datasets

# Load the iris dataset and split it into features and target
# Visualize graphically the dataset

iris = sklearn.datasets.load_iris()
ax = matplotlib.pyplot.subplot()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend(scatter.legend_elements()[0], iris.target_names, title="Classes")
matplotlib.pyplot.show()


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
classifier = sklearn.tree.DecisionTreeClassifier(max_depth=3, random_state=42)
classifier = classifier.fit(X_train, y_train)
sample_flower = [[5.0, 1.5, 0.5, 3.5]]
predicted_class = classifier.predict(sample_flower)
print(f"The predicted class for the flower with features {sample_flower} is: {iris.target_names[predicted_class][0]}")

# Visualize the decision tree
dot_data = io.StringIO()
sklearn.tree.export_graphviz(classifier, out_file=dot_data, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,
                     filled=True, rounded=True,  
                     special_characters=True)
dot_data = dot_data.getvalue()
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # Saves the tree as a PDF file
graph.view()  # Opens the PDF file

