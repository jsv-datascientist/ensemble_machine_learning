from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

#  Load the wine dataset and assign the features and labels to X and y
X, y = load_wine(return_X_y=True)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the RandomForestClassifier with 100 estimators and max_depth of 3
random_clf = RandomForestClassifier(n_estimators=100 , max_depth=3)
random_clf.fit(X_train, y_train)

#  Make predictions and calculate accuracy
y_pred = random_clf.predict(X_test)

#  Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is {accuracy}")