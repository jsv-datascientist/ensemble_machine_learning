from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load digit dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining base and meta models, this is a tuple lisy
estimators = [
    ('rf', RandomForestClassifier ()),
    ('svc', SVC(random_state=42))
]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42))
stack_clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = stack_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Classifier Accuracy: {accuracy:.2f}")