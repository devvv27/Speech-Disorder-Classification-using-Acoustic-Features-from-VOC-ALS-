import pandas as pd
import subprocess
import sys

print("Running all baseline models...\n")

models = [
    ("Logistic Regression", "train_logistic_regression.py", "logistic_regression_results.csv"),
    ("SVM (RBF)", "train_svm.py", "svm_rbf_results.csv"),
    ("Random Forest", "train_random_forest.py", "random_forest_results.csv"),
]

results_list = []

for name, script, csv_file in models:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], cwd=".")
    if result.returncode == 0:
        df = pd.read_csv(csv_file)
        results_list.append(df)
        print(f"✓ {name} completed successfully")
    else:
        print(f"✗ {name} failed")

if results_list:
    summary_df = pd.concat(results_list, ignore_index=True)
    summary_df.to_csv("baseline_summary_results.csv", index=False)
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nBest model by F1 Score: {summary_df.loc[summary_df['f1'].idxmax(), 'model']}")
    print(f"Summary saved to baseline_summary_results.csv")
else:
    print("No models completed successfully.")
