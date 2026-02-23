#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import json
import gzip 
import csv
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path("..")
DATA_DIR = BASE_DIR / "data"

FILE_PATH = DATA_DIR / "openfoodfacts-products.jsonl.gz"

OUT_CSV = BASE_DIR / "gen" / "off_selected_fields_filtered.csv"
OUT_TXT = BASE_DIR / "gen" / "openfoodfacts_10records.txt"


# In[5]:


# Save first N full records to a .txt file

def save_full_first_n(path: Path, n: int, out_file: Path):
    """
    Saves the first N records from a gzipped JSONL file.
    out_file MUST be explicitly provided.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, 'rt', encoding='utf-8') as f, out_file.open("w", encoding="utf-8") as out:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            out.write(f"\n— FULL RECORD #{i} —\n")
            out.write(json.dumps(obj, ensure_ascii=False, indent=2))
            out.write("\n\n")
            if i >= n:
                break

    print(f"Saved {n} records to {out_file.resolve()}")

save_full_first_n(FILE_PATH, n=10, out_file=OUT_TXT)


# In[7]:


# Only take complete observations

def is_empty(v):
    return v is None or v == "" or v == [] or v == {}

def first_non_empty(*vals):
    for v in vals:
        if not is_empty(v):
            return v
    return ""

def normalize(val):
    if is_empty(val):
        return ""
    if isinstance(val, list):
        return " | ".join(str(x) for x in val)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def nutriscore_2023_valid(value):
    """Return True if nutriscore_2023_tags contains at least one tag
    that is not 'unknown' or 'not-applicable' (case-insensitive)."""
    if is_empty(value):
        return False
    if isinstance(value, str):
        tags = [value]
    elif isinstance(value, list):
        tags = [str(x) for x in value]
    else:
        return False
    banned = {"unknown", "not-applicable"}
    # keep any tag that isn't empty and not banned
    kept = [t for t in (s.strip().lower() for s in tags) if t and t not in banned]
    return len(kept) > 0

# count for ETA
with gzip.open(FILE_PATH, 'rt', encoding='utf-8') as f:
    total = sum(1 for _ in f)

fields = [
    "barcode",
    "generic_name_en",      
    "labels_tags",
    "nutriscore_2021_tags",
    "nutriscore_2023_tags",
    "nutriscore_data",
    "nutriscore_score",
    "nova_group",
    "nova_groups_markers",
    "brands",
    "brand_owner",
    "categories_hierarchy", 
    "countries_hierarchy",
    "additives_n",
    "additives_original_tags",
    "ingredients_analysis",
    "ingredients_analysis_tags",
    "ingredients_original_tags",
    "ingredients_n",
    "known_ingredients_n",
    "nutrient_levels",
    "nutriments",
    "stores_tags",
]

kept = 0
with gzip.open(FILE_PATH, 'rt', encoding='utf-8') as f, open(OUT_CSV, "w", newline="", encoding="utf-8") as out:
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()

    for line in tqdm(f, total=total, desc="Filtering & exporting"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Required fields 
        if is_empty(obj.get("labels_tags")):
            continue
        if not nutriscore_2023_valid(obj.get("nutriscore_2023_tags")):
            continue
        if is_empty(obj.get("nova_group")):
            continue

        # Build rows
        # barcode
        barcode = first_non_empty(obj.get("_id"), obj.get("id"), obj.get("code"))

        # name with fallbacks
        name = first_non_empty(
            obj.get("generic_name_en"),
            obj.get("product_name_en"),
            obj.get("ciqual_food_name:en"),
            obj.get("product_name"),
        )

        # categories_hierarchy with 'undefined' handling -> fallback to categories_old
        cat_h = obj.get("categories_hierarchy")
        use_cat_old = False
        if is_empty(cat_h):
            use_cat_old = True
        else:
            if isinstance(cat_h, list) and any(str(x).lower().endswith("undefined") for x in cat_h):
                use_cat_old = True
            elif isinstance(cat_h, str) and "undefined" in cat_h.lower():
                use_cat_old = True
        if use_cat_old:
            cat_h = obj.get("categories_old")

        row = {
            "barcode": normalize(barcode),
            "generic_name_en": normalize(name),
            "labels_tags": normalize(obj.get("labels_tags")),
            "nutriscore_2021_tags": normalize(obj.get("nutriscore_2021_tags")),
            "nutriscore_2023_tags": normalize(obj.get("nutriscore_2023_tags")),
            "nutriscore_data": normalize(obj.get("nutriscore_data")),
            "nutriscore_score": normalize(obj.get("nutriscore_score")),
            "nova_group": normalize(obj.get("nova_group")),
            "nova_groups_markers": normalize(obj.get("nova_groups_markers")),
            "brands": normalize(obj.get("brands")),
            "brand_owner": normalize(obj.get("brand_owner")),
            "categories_hierarchy": normalize(cat_h),
            "countries_hierarchy": normalize(obj.get("countries_hierarchy")),
            "additives_n": normalize(obj.get("additives_n")),
            "additives_original_tags": normalize(obj.get("additives_original_tags")),
            "ingredients_analysis": normalize(obj.get("ingredients_analysis")),
            "ingredients_analysis_tags": normalize(obj.get("ingredients_analysis_tags")),
            "ingredients_original_tags": normalize(obj.get("ingredients_original_tags")),
            "ingredients_n": normalize(obj.get("ingredients_n")),
            "known_ingredients_n": normalize(obj.get("known_ingredients_n")),
            "nutrient_levels": normalize(obj.get("nutrient_levels")),
            "nutriments": normalize(obj.get("nutriments")),
            "stores_tags": normalize(obj.get("stores_tags")),
        }

        w.writerow(row)
        kept += 1

print(f"Done. Kept {kept:,} products. Wrote: {OUT_CSV.resolve()}")

