import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    
    return X_scaled, y, X.columns

def main():
    X, y, feature_names = load_and_prepare_data("voc_als_acoustic_features_mapped.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    print("="*60)
    print("SVM (RBF Kernel) BASELINE")
    print("="*60)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Class distribution (test): {pd.Series(y_test).value_counts().to_dict()}")
    
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("SVM (RBF) Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("svm_rbf_confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix plot to svm_rbf_confusion_matrix.png")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"  Cross-Val F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save model
    joblib.dump(model, "svm_rbf_model.pkl")
    print("Saved model to svm_rbf_model.pkl")
    
    # Save results
    results_df = pd.DataFrame({
        "model": ["SVM (RBF)"],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1": [f1],
        "cv_f1_mean": [cv_scores.mean()],
        "cv_f1_std": [cv_scores.std()]
    })
    results_df.to_csv("svm_rbf_results.csv", index=False)
    print("Saved results to svm_rbf_results.csv")
    print("="*60)

if __name__ == "__main__":
    main()
