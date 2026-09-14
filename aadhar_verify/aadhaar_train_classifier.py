import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES = ["ela", "edge", "noise_texture", "sharpness"]


def load_dataset(path="aadhaar_dataset.csv"):
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[k]) for k in FEATURES])
            y.append(int(row["label"]))
    return np.array(X), np.array(y)


def train(path="aadhaar_dataset.csv", model_out="aadhaar_tamper_model.joblib", model_type="rf"):
    X, y = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["genuine", "tampered"]))

    cv_scores = cross_val_score(clf, X, y, cv=5)
    print("5-fold CV accuracy: %.3f +/- %.3f  (more reliable than the single test-split number above)"
          % (cv_scores.mean(), cv_scores.std()))

    if model_type == "rf":
        print("\nFeature importances:")
        for name, imp in zip(FEATURES, clf.feature_importances_):
            print(f"  {name:15s}: {imp:.3f}")
    else:
        print("\nLearned feature weights:")
        for name, coef in zip(FEATURES, clf.coef_[0]):
            print(f"  {name:15s}: {coef:+.3f}")

    joblib.dump(clf, model_out)
    print(f"\nModel saved to {model_out}")
    return clf


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "rf"
    train(model_type=model_type)
