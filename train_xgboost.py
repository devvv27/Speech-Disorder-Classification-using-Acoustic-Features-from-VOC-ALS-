import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop non-numeric and metadata columns
    drop_cols = ["label", "file_id", "subject_id", "Sex", "Age", "Severity", "Severity_imputed"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels to numeric (0..n_classes-1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    return X_scaled, y_encoded, X.columns, le

def main():
    X, y, feature_names, le = load_and_prepare_data("voc_als_acoustic_features_mapped.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("=" * 60)
    print("XGBOOST CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution (train): {pd.Series(le.inverse_transform(y_train)).value_counts().to_dict()}")
    print(f"Class distribution (test): {pd.Series(le.inverse_transform(y_test)).value_counts().to_dict()}")
    print(f"Number of features: {X.shape[1]}")
    print()
    
    # Train XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    # Save the label encoder so predictions can be mapped back to original labels
    joblib.dump(le, "label_encoder.pkl")
    print("✓ Model training complete")
    print()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print("Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    # Show original string labels on axes
    label_names = le.inverse_transform(model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("XGBoost Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("xgboost_confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix plot to xgboost_confusion_matrix.png")
    print()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"Cross-Val F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    
    # Save model
    joblib.dump(model, "xgboost_model.pkl")
    print("Saved model to xgboost_model.pkl")
    
    # Save results
    results_df = pd.DataFrame({
        "model": ["XGBoost"],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1": [f1],
        "cv_f1_mean": [cv_scores.mean()],
        "cv_f1_std": [cv_scores.std()],
        "num_features": [X.shape[1]]
    })
    results_df.to_csv("xgboost_results.csv", index=False)
    print("Saved results to xgboost_results.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
