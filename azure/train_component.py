import argparse
import json
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ready", type=str, required=True)
    parser.add_argument("--test_ready", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="house_affiliation")
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    print("="*60)
    print("Decision Tree Training Component")
    print("="*60)

    print("\n1. Loading data...")
    X_train_path = os.path.join(args.train_ready, "X_train.csv")
    y_train_path = os.path.join(args.train_ready, "y_train.csv")
    X_test_path = os.path.join(args.test_ready, "X_test.csv")
    y_test_path = os.path.join(args.test_ready, "y_test.csv")

    for p in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[args.target_col].astype(str)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[args.target_col].astype(str)

    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples")

    mlflow.start_run()

    print("\n2. Training Decision Tree Classifier...")
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("target_col", args.target_col)

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train)
    print("   ✓ Model trained")

    print("\n3. Evaluating model...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc = float(accuracy_score(y_test, y_pred_test))
    
    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"{label}_precision", metrics['precision'])
            mlflow.log_metric(f"{label}_recall", metrics['recall'])
            mlflow.log_metric(f"{label}_f1", metrics['f1-score'])

    print("\n4. Saving model...")
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(clf, model_path)
    print(f"   Model saved: {model_path}")

    # Log model to MLflow (but don't register yet - that's done in separate step)
    mlflow.sklearn.log_model(clf, "decision_tree_model")

    report_path = os.path.join(args.model_output, "metrics.json")
    with open(report_path, 'w') as f:
        json.dump({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "classification_report": report
        }, f, indent=2)

    mlflow.end_run()

    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
