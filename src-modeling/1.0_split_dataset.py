#!/usr/bin/env python
# coding: utf-8

# # Split dataset
# Split in 70/30 Training & Test sets. 

# In[1]:


import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ## 1. Read and Check dataset

# In[ ]:


df = pd.read_csv("results/data_def.csv")

print("Shape:")
print(df.shape)
print("------------------------------------------------------")

print("Columns:")
print(df.columns)
print("------------------------------------------------------")

print("Info:")
print(df.info())


# In[13]:


df.sample(n=25, random_state=42)


# ## 2. Functions for check splits

# In[15]:


# Create main_mapped_category based on the prefix number in mapped_category
def get_main_category(subcat):
    if pd.isna(subcat):
        return None
    prefix = str(subcat).split(".")[0]
    mapping = {
        "1": "1. Beverages",
        "2": "2. Fruits & Vegetables",
        "3": "3. Cereals & Starches",
        "4": "4. Dairy",
        "5": "5. Meat, Fish & Eggs",
        "6": "6. Fats & Sauces",
        "7": "7. Composite & Prepared Meals",
        "8": "8. Snacks & Appetisers",
        "9": "9. Sweet Products & Desserts"
    }
    return mapping.get(prefix, "Unknown")

df["main_mapped_category"] = df["mapped_category"].apply(get_main_category)


# In[17]:


# Label coverage tiers
def label_tier(row):
    n_groups = sum([
        row["n_1_intrinsic"] > 0,
        row["n_2_extrinsic"] > 0,
        row["n_3_packaging"] > 0,
    ])
    if n_groups == 0:
        return "none"
    elif n_groups == 1:
        return "1"
    elif n_groups == 2:
        return "2"
    else:
        return "3"

df["label_tier"] = df.apply(label_tier, axis=1)


# In[19]:


df.sample(n=25, random_state=42)


# ## 3. Create Stratifying variable (category x nova)

# In[ ]:


df["stratify_raw"] = df["main_mapped_category"].astype(str) + " | " + df["nova_group"].astype(str)
counts = df["stratify_raw"].value_counts().reset_index()
counts.columns = ["stratify_raw", "count"]
total = counts["count"].sum()
counts["percent"] = (counts["count"] / total * 100).round(2)

print(counts)
counts.to_csv("gen/main_stratify_counts.csv", index=False)


# In[25]:


# Sort descending by count
counts_sorted = counts.sort_values("percent", ascending=True)

plt.figure(figsize=(8, 18))  # taller for 120 categories
plt.barh(counts_sorted["stratify_raw"], counts_sorted["percent"], color="steelblue", edgecolor="grey")

plt.xlabel("Share of total dataset (%)")
plt.ylabel("Stratification group (category × NOVA)")
plt.title("Distribution of category × NOVA strata before thresholding")

# line for threshold
plt.axvline(x=0.25, color="red", linestyle="--", label="0.25% threshold")
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


THRESHOLD = 0.25  # percent of total
rare_stratify = counts.loc[counts["percent"] < THRESHOLD, "stratify_raw"]
df["stratify"] = df["stratify_raw"].where(~df["stratify_raw"].isin(rare_stratify), "0.0 Rare")

stratify = df["stratify"].value_counts().reset_index()
stratify.columns = ["stratify", "count"]

total = stratify["count"].sum()
stratify["percent"] = (stratify["count"] / total * 100).round(2)

print(stratify)
stratify.to_csv("gen/main_stratify_counts_after_threshold.csv", index=False)


# ## 4. Global Split data (70/30)

# In[29]:


train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.30,
    random_state=0,
    stratify=df["stratify"]
)

df.loc[train_idx, "split"] = "train"
df.loc[test_idx,  "split"] = "test"

print(df["split"].value_counts(normalize=True).round(3))
df.sample(n=25, random_state=0)


# ## 5. Checks

# In[31]:


def pct_table(s):
    """Return % distribution table for a Series"""
    return (s.value_counts(normalize=True)
              .mul(100)
              .round(2)
              .rename("overall_percent")
              .to_frame())

def compare_split(df, col):
    """Compare overall / train / test distributions side-by-side"""
    overall = (df[col].value_counts(normalize=True) * 100).round(2)
    train   = (df.loc[df.split=="train", col].value_counts(normalize=True) * 100).round(2)
    test    = (df.loc[df.split=="test",  col].value_counts(normalize=True) * 100).round(2)
    
    comp = pd.concat([overall, train, test], axis=1, sort=True)
    comp.columns = ["overall_%", "train_%", "test_%"]
    comp = comp.fillna(0).sort_values("overall_%", ascending=False)
    comp = comp.sort_index()

    return comp

# --- Variables to check ---
cols_to_check = ["main_mapped_category", "nova_group", "nutriscore", "label_tier", "stratify"]

for col in cols_to_check:
    comp = compare_split(df, col)
    print(f"\n=== {col} ===")
    print(comp.to_string())


# ## 6. Create Final Datasets

# In[ ]:


df.loc[df.split=="train"].to_csv("gen/data_def_train.csv", index=False)
df.loc[df.split=="test"].to_csv("gen/data_def_test.csv", index=False)

