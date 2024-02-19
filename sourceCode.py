from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set randomness based on a seed, meaning if used the results can be the same after every code run
# import numpy as np
# np.random.seed(65)

# read the dataset from the file and display it
dataset = pd.read_csv("vertebral_column.dat", skiprows=0)
print(dataset)

# use the K Nearest Neighbour Classifier
classifier = KNeighborsClassifier(n_neighbors=7)
# or use the Multi Layer Perceptron Classifier
# classifier = MLPClassifier()

# uses every row and the first 6 columns as attributes
X = dataset.values[:, 0:5]
# uses every row and the 7th column as the label data
Y = dataset.values[:, 6]

# trains and tests the dataset with a test set of size 20% of the full dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# The following code implements feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# uses the classifier on the dataset
classifier = classifier.fit(X_train, Y_train)
# the system predicts results using the test set
Y_prediction = classifier.predict(X_test)

# The following code prints the prediction made compared to the actual results (O for correct, X for incorrect)
print("Start of prediction")
for i in range(0, len(Y_prediction)):
    print(Y_prediction[i] + " | " + Y_test[i] + " -> " + ("O" if Y_prediction[i] == Y_test[i] else "X"))
print("End of prediction")

# prints the accuracy of the prediction compared to the actual results
print("Train/test accuracy:", accuracy_score(Y_test, Y_prediction))

# use cross validation partitioning the data into 5 sub-samples each 20% the entire data
crossValidation = ShuffleSplit(n_splits=5, test_size=0.2)
scores = cross_val_score(classifier, X, Y, cv=crossValidation)
# prints the scores and mean value of the cross validation tests
print("Cross fold validation accuracy scores:", scores)
print("Cross fold validation accuracy mean:", scores.mean())

# prints the precision
precision = precision_score(Y_test, Y_prediction, average="macro")
print("Precision: ", precision)
