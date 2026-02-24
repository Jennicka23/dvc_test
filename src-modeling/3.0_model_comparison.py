#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import json

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

leaderboard = comparison_df.set_index('Model')['BalAcc'].to_dict()

# Save as JSON with indentation for a perfect vertical view
with open("results/final_model_comparison.json", "w") as f:
    json.dump(leaderboard, f, indent=4)

print("\nJSON saved for DVC metrics.")

