from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train an AdaBoost classifier with 50 estimators, fit to training data, and evaluate accuracy
ada_clf = AdaBoostClassifier(n_estimators=50)
ada_clf.fit(X_train, y_train)
y_predict = ada_clf.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_predict)

#  Train a Gradient Boosting classifier with 50 estimators, fit to training data, and evaluate accuracy
gb_clf = GradientBoostingClassifier(n_estimators=50)
gb_clf.fit(X_train, y_train)
y_predict = gb_clf.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_predict)

# Print accuracies
print(f"Accuracy for AdaBoost: {accuracy_ada:.2f}")
print(f"Accuracy for Gradient Boosting: {accuracy_gb:.2f}")