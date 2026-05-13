import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_CSV = "voc_als_acoustic_features_mapped.csv"
RF_IMPORTANCE_CSV = "random_forest_feature_importance.csv"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    drop_cols = ["label", "file_id", "subject_id", "Sex", "Age", "Severity", "Severity_imputed"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    y = df["label"].astype(str)
    feature_names = X.columns.tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, feature_names, le


def load_top_features(importance_csv, top_n):
    importance_df = pd.read_csv(importance_csv)
    required_cols = {"feature", "importance"}
    if not required_cols.issubset(importance_df.columns):
        raise ValueError(f"{importance_csv} must contain columns: {sorted(required_cols)}")

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    top_features = importance_df.head(top_n)["feature"].tolist()
    return importance_df, top_features


def select_features(X, feature_names, top_features):
    available_features = [feature for feature in top_features if feature in feature_names]
    if not available_features:
        raise ValueError("None of the requested top features were found in the dataset.")

    selected = X[available_features].copy()
    scaler = StandardScaler()
    selected_scaled = scaler.fit_transform(selected)
    return selected_scaled, available_features, scaler


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, class_names, output_prefix):
    print("=" * 70)
    print(f"{model_name} TRAINING")
    print("=" * 70)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=class_names)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix plot to {output_prefix}_confusion_matrix.png")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"Cross-Val F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "confusion_matrix_file": f"{output_prefix}_confusion_matrix.png",
    }


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression and XGBoost using RF top features.")
    parser.add_argument("--top_n", type=int, default=30, help="Number of top RF features to keep.")
    args = parser.parse_args()

    X_df, y, feature_names, le = load_dataset(DATA_CSV)
    importance_df, top_features = load_top_features(RF_IMPORTANCE_CSV, args.top_n)

    X_selected, selected_features, scaler = select_features(X_df, feature_names, top_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    class_names = list(le.classes_)

    print("=" * 70)
    print("TRAINING ON TOP RANDOM-FOREST-RANKED FEATURES")
    print("=" * 70)
    print(f"Dataset: {DATA_CSV}")
    print(f"RF importance source: {RF_IMPORTANCE_CSV}")
    print(f"Requested top features: {args.top_n}")
    print(f"Selected features used: {len(selected_features)}")
    print(f"Class distribution: {pd.Series(le.inverse_transform(y)).value_counts().to_dict()}")
    print()
    print("Selected features and importances:")
    for idx, feature in enumerate(selected_features, start=1):
        importance_value = importance_df.loc[importance_df["feature"] == feature, "importance"].iloc[0]
        print(f"  {idx:>2}. {feature} ({importance_value:.6f})")
    print()

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_results = evaluate_model(
        model_name="Logistic Regression",
        model=lr_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        output_prefix="lr_top_features",
    )
    joblib.dump(lr_model, "logistic_regression_top_features_model.pkl")
    print("Saved model to logistic_regression_top_features_model.pkl")
    print()

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_results = evaluate_model(
        model_name="XGBoost",
        model=xgb_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        output_prefix="xgboost_top_features",
    )
    joblib.dump(xgb_model, "xgboost_top_features_model.pkl")
    print("Saved model to xgboost_top_features_model.pkl")
    print()

    results_df = pd.DataFrame([
        {**lr_results, "num_features": len(selected_features), "top_n_requested": args.top_n, "feature_source": RF_IMPORTANCE_CSV},
        {**xgb_results, "num_features": len(selected_features), "top_n_requested": args.top_n, "feature_source": RF_IMPORTANCE_CSV},
    ])
    results_df.to_csv("lr_xgboost_top_features_results.csv", index=False)
    print("Saved results to lr_xgboost_top_features_results.csv")

    pd.DataFrame({"feature": selected_features}).to_csv("selected_top_rf_features.csv", index=False)
    importance_df.to_csv("random_forest_feature_importance_sorted.csv", index=False)
    scaler_info = pd.DataFrame({"feature": selected_features})
    scaler_info.to_csv("selected_top_rf_features_order.csv", index=False)
    joblib.dump(scaler, "top_features_scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Saved selected features to selected_top_rf_features.csv")
    print("Saved sorted RF importance to random_forest_feature_importance_sorted.csv")
    print("Saved scaler to top_features_scaler.pkl")
    print("Saved label encoder to label_encoder.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
