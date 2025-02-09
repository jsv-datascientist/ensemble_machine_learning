from sklearn.datasets import load_digits
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset and split it
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=12, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=6, random_state=42))
]

# List of possible meta models
meta_models = [
    LogisticRegression(max_iter=100),
    RidgeClassifier(),
    DecisionTreeClassifier()
]

# Iterate through meta models and print accuracy
for meta_model in meta_models:
    #  train StackingClassfier with estimators=estimators and final_estimator=meta_model
    model = StackingClassifier(estimators= estimators, final_estimator=meta_model)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_predict)
    #  make predictions and calculate the accuracy. Put the accuracy into the accuracy variable
    print(f'Meta Model: {meta_model.__class__.__name__}, Accuracy: {accuracy:.2f}')