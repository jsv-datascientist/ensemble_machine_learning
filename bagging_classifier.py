from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_accuracy = 0
best_n_estimators = 0

for n in range(10, 110, 10):
    #  Initialize a BaggingClassifier with DecisionTreeClassifier and given n_estimators (n) and put it into the bag_clf variable
    #  Fit the BaggingClassifier to the training data
    bag_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators= n)
    bag_clf.fit(X_train, y_train)
    
    y_pred = bag_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_estimators = n

print(f"Best n_estimators: {best_n_estimators}, Best Accuracy: {best_accuracy:.2f}")