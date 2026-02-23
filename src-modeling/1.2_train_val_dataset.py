#!/usr/bin/env python
# coding: utf-8

# # Train/Val dataset
# Create 5-fold stratified cross-validation

# In[11]:


import pandas as pd
from sklearn.model_selection import StratifiedKFold


# In[13]:


df_train = pd.read_csv("gen/data_def_train.csv")

print("Training Dataset size:", len(df_train))
df_train.head()


# In[20]:


# Stratified 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
targets = ["nova_group", "nutriscore"]

fold_assignments = pd.DataFrame({"index": df_train.index})

for target in targets:
    fold_col = f"{target}_fold"
    fold_assignments[fold_col] = -1  # placeholder

    for fold, (train_idx, val_idx) in enumerate(cv.split(df_train, df_train[target])):
        fold_assignments.loc[val_idx, fold_col] = fold + 1

df_train = pd.concat([df_train, fold_assignments.drop(columns="index")], axis=1)


# In[22]:


for target in targets:
    print(f"\n=== Stratified 5-Fold distributions for {target} ===")

    # collect counts per fold
    dist_df = pd.DataFrame()
    for i in range(1, 6):
        subset = df_train[df_train[f"{target}_fold"] == i]
        dist = subset[target].value_counts(normalize=True).round(3)
        dist_df[f"Fold {i}"] = dist

    # sort order: numeric (1>4) or alphabetic (A>E)
    if target == "nova_group":
        dist_df = dist_df.sort_index(key=lambda x: x.astype(int))
    else:
        dist_df = dist_df.sort_index(key=lambda x: x.str.upper())

    # add total column (average across folds)
    dist_df["Mean"] = dist_df.mean(axis=1).round(3)

    print(dist_df.to_string())


# In[24]:


print("=== Dataset Overview ===")
print(f"Total rows in training dataset: {len(df_train):,}")
print(f"Columns: {', '.join(df_train.columns[:10])} ...")  
print("\nFirst 10 rows:")
print(df_train.head(10))  

print("\n=== Fold sizes ===")
for target in targets:
    print(f"\nTarget: {target}")
    fold_sizes = df_train[f"{target}_fold"].value_counts().sort_index()
    for fold, size in fold_sizes.items():
        print(f"  Fold {fold}: {size:,} rows ({size/len(df_train)*100:.2f} %)")
    print(f"  Mean per fold: {fold_sizes.mean():,.0f} rows\n")


# In[28]:


# Collect fold size stats for all targets
fold_summary = []

for target in targets:
    fold_sizes = df_train[f"{target}_fold"].value_counts().sort_index()
    stats = fold_sizes.describe().round(2).to_dict()

    fold_summary.append({
        "target": target,
        "total_rows": len(df_train),
        "mean_fold_size": stats["mean"],
        "std_fold_size": stats["std"],
        "min_fold_size": stats["min"],
        "max_fold_size": stats["max"],
        "cv_percent": round(stats["std"] / stats["mean"] * 100, 3),
    })

fold_summary_df = pd.DataFrame(fold_summary)

print("\n=== Fold Size Summary ===")
print(fold_summary_df.to_string(index=False))


# In[30]:


df_train.to_csv("gen/data_def_train_folds.csv", index=False)

