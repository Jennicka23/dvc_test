#!/usr/bin/env python
# coding: utf-8

# # Test dataset
# Add core_slice flag to test dataset

# In[6]:


import pandas as pd


# In[8]:


df_test  = pd.read_csv("gen/data_def_test.csv")

# Define core slice
df_test["core_slice"] = (
    (df_test["n_1_intrinsic"] > 0) &
    (df_test["n_2_extrinsic"] > 0) &
    (df_test["n_3_packaging"] > 0)
)

# Check proportions
core_share = df_test["core_slice"].mean() * 100
core_n = df_test["core_slice"].sum()
print(f"Core slice size: {core_n:,} items ({core_share:.2f}% of test set)")

# save for reference
df_test.to_csv("gen/data_def_test.csv", index=False)
print("Updated data_def_test.csv with 'core_slice' flag.")


# In[10]:


df_test.head()

