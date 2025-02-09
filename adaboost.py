from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_wine(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost classifier with DecisionTreeClassifier as the base estimator
dt_clf = RandomForestClassifier ()
ada_clf = AdaBoostClassifier(estimator=dt_clf, n_estimators=100, algorithm='SAMME')
ada_clf.fit(X_train, y_train)

# Make predictions
y_pred_ada = ada_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_ada)
print(f"AdaBoost Classifier accuracy: {accuracy}")