"""
================================================================================
Model Trainer — Train 13 ML models including ensembles
================================================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def build_models(random_state: int = 42) -> dict:
    """
    Returns a dict of name → untrained estimator for every model
    in the comparison, including two voting ensembles.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, solver="lbfgs", random_state=random_state, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=3,
            min_samples_split=5, random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4,
            min_samples_split=2, random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=4, random_state=random_state,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True, random_state=random_state,
        ),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50, random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=random_state,
        ),
        "Dense Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500, random_state=random_state,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            random_state=random_state, eval_metric="logloss",
        ),
    }

    # Voting Ensemble 1 — all models (soft voting)
    all_estimators = [(name, m) for name, m in models.items()]
    models["Voting Ensemble 1"] = VotingClassifier(
        estimators=all_estimators, voting="soft"
    )

    # Voting Ensemble 2 — top-3 tree boosters (soft voting)
    best_3 = [
        ("XGBoost", models["XGBoost"]),
        ("Gradient Boosting", models["Gradient Boosting"]),
        ("Random Forest", models["Random Forest"]),
    ]
    models["Voting Ensemble 2"] = VotingClassifier(
        estimators=best_3, voting="soft"
    )

    return models


def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """
    Builds and trains every model on the given training set.

    Returns:
        dict: name → fitted model.
    """
    models = build_models(random_state)

    print("\n==========================================================================")
    print(f"               TRAINING {len(models)} MACHINE LEARNING MODELS")
    print("==========================================================================")

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)

    print(f"[SUCCESS] All {len(models)} models trained successfully.")
    return models
