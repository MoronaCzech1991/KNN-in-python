# Import the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Load the data
dataSet = pd.read_csv("heart.csv")

# Split the dataset
X = dataSet.iloc[:, 0:13]
y = dataSet.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

# Scaling the data
scalingData = StandardScaler()
X_train = scalingData.fit_transform(X_train)
X_test = scalingData.transform(X_test)

# Create the model
classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, metric = "euclidean")
classifier.fit(X_train, y_train)

# Evaluate the model
y_predict = classifier.predict(X_test)

print(f1_score(y_test, y_predict))
print(accuracy_score(y_test, y_predict))

# Using the confusion
confusion_matrix = metrics.confusion_matrix(y_test, classifier.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
