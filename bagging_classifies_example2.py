from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset and split into training and testing sets
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a bagging classifier with different numbers of estimators and different base models
best_accuracy = 0
best_n_estimators = 0
best_base_model = None
base_models = [DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB()]

for model in base_models:
    for n in range(50, 151, 10):
        #  train bagging classifier with parameters estimator=model and n_estimators=n
        #  calculate the accuracy on the testing data and put it into the bag_accuracy variable
        bag_clf = BaggingClassifier(estimator= model, n_estimators=n )
        bag_clf.fit(X_train, y_train)
        
        bag_accuracy = accuracy_score(y_test, bag_clf.predict(X_test))
        if bag_accuracy > best_accuracy:
            best_accuracy = bag_accuracy
            best_n_estimators = n
            best_base_model = model.__class__.__name__

print(f"Best accuracy achieved: {best_accuracy:.2f} with {best_n_estimators} n_estimators and {best_base_model} as the base model")