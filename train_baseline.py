import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=["label", "file_id"])
    y = df["label"]
    
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ALS", "HC"], yticklabels=["ALS", "HC"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"  Cross-Val F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    X, y, feature_names = load_and_prepare_data("voc_als_acoustic_features.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Class distribution (test): {pd.Series(y_test).value_counts().to_dict()}")
    
    results = []
    
    results.append(train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)))
    results.append(train_and_evaluate(X_train, X_test, y_train, y_test, "SVM", SVC(kernel="rbf", random_state=42)))
    results.append(train_and_evaluate(X_train, X_test, y_train, y_test, "Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("baseline_results.csv", index=False)
    print("\n" + "="*50)
    print("Summary Results:")
    print(results_df.to_string(index=False))
    print(f"\nBest Model by F1 Score: {results_df.loc[results_df['f1'].idxmax(), 'model']}")

if __name__ == "__main__":
    main()
