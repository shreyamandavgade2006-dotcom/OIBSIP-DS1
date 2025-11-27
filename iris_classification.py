from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset from sklearn
iris = load_iris()

# Features (measurements)
X = iris.data  # sepal length, sepal width, petal length, petal width

# Target labels (species)
y = iris.target  # 0: setosa, 1: versicolor, 2: virginica

print("Dataset Loaded Successfully")
print("\nFeature sample:\n", X[:5])
print("\nTarget sample:", y[:5])

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict accuracy on test data
y_pred = model.predict(X_test)
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# User input for prediction
print("\nEnter Iris flower measurements")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Create input array
new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict species
prediction = model.predict(new_sample)[0]
species = iris.target_names[prediction]

print("\nPredicted Species:", species)

#output
#Sepal length (cm): 6.3
#Sepal width (cm): 3.0
#Petal length (cm): 5.9
#Petal width (cm): 2.0

#Predicted Species: virginica

