from sklearn import datasets, model_selection, tree

import matplotlib.pyplot as plt
import graphviz
import io

iris = datasets.load_iris()
ax = plt.subplot()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend(scatter.legend_elements()[0], iris.target_names, title="Classes")
plt.show()


X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
classifier = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
classifier = classifier.fit(X_train, y_train)
sample_flower = [[5.0, 3.5, 1.5, 0.5]]
predicted_class = classifier.predict(sample_flower)
print(f"The predicted class for the flower with features {sample_flower[0]} is: {iris.target_names[predicted_class][0]}")

# mostra a arvore
dot_data = io.StringIO()
tree.export_graphviz(classifier, out_file=dot_data, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,
                     filled=True, rounded=True,  
                     special_characters=True)
dot_data = dot_data.getvalue()
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")
graph.view()