#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

results_files = {
    "Nutri-Baseline": "results/logreg_nutri_eval_metrics_train_test_core.csv",
    "NOVA-Baseline": "results/logreg_nova_eval_metrics_train_test_core.csv",
    "Nutri-RF": "results/rf_nutri_eval_metrics_train_test_core.csv",
    "NOVA-RF": "results/rf_nova_eval_metrics_train_test_core.csv"
}

summary_list = []

for model_name, path in results_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)

        # Grab only Phase 3 (all labels) on the Test-Core slice
        row = df[(df["Phase"] == "phase3_all_labels") & 
                 (df["Dataset"] == "Test-Core")].copy()
        
        if not row.empty:
            summary_list.append({
                "Model": model_name,
                "BalAcc": row["Balanced Accuracy"].values[0]
            })

# Create the clean dataframe
comparison_df = pd.DataFrame(summary_list)

# Sort so the best model is at the top
comparison_df = comparison_df.sort_values("BalAcc", ascending=False)

# Save with standard comma (NO SEMICOLONS) for DVC
comparison_df.to_csv("results/final_model_comparison_table.csv", index=False)

print("\n--- Clean Comparison (Phase 3 Test-Core) ---")
print(comparison_df.to_string(index=False))

