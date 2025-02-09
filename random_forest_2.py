import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the wine dataset
X, y = load_wine(return_X_y=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Store accuracies for different max_depth values
depths = list(range(1, 15, 1))
accuracies = []

# Evaluate RandomForestClassifier for each max_depth
for depth in depths:
    #  train the RandomForestClassifier with max_depth=depth and 100 estimators
    #  put the predictions on the testing set to the y_pred variable
    random_clf = RandomForestClassifier(n_estimators=100, max_depth=depth)
    random_clf.fit(X_train, y_train)
    y_pred = random_clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs max_depth
plt.plot(depths, accuracies, marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy vs max_depth')
plt.grid(True)
plt.show()