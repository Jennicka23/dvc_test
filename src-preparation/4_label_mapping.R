# ---------------------------- 1. Setup ---------------------------------------#
# Setup the environment:
rm(list=ls())
gc()
cat("\014")
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(readxl, dplyr, tidyr, purrr, tibble, readr, stringr, here, jsonlite, ggplot2, widyr, scales,tidyverse, reshape2, Matrix, fastcluster, coop, dendextend, RColorBrewer, igraph, knitr)

# ---------------------------- 2. Input ---------------------------------------#

category_mapped_filtered <- read_csv("gen/category_mapped_filtered.csv")
labels_long <- read_csv("gen/labels_long.csv")
labels_mapping <- read_excel("data/labels_mapping.xlsx")

# ------------------------- 3. Transformation  --------------------------------#

# Merge label mapping with labels_long
labels_long <- labels_long %>%
  left_join(
    labels_mapping %>%
      select(label, Canonical2, Category, Subcategory),
    by = "label"
  )

labels_def <- labels_long %>%
  # Remove unmapped labels
  filter(!(is.na(Canonical2) & is.na(Category) & is.na(Subcategory))) %>%
  # Keep distinct Canonical2 values per barcode
  distinct(barcode, Canonical2, .keep_all = TRUE) %>% 
  # Exclude unwanted labels
  filter(
    # remove any Canonical2 starting with "nutriscore"
    !str_detect(str_to_lower(Canonical2), "^nutriscore"),
    # remove exact matches (ignore whitespace/newlines)
    !str_trim(Canonical2) %in% c("INCORRECT_NUTRITION_DATA", "INCORRECT_LABEL_DATA")
  )

# Collapse Canonical2 per barcode + count how many distinct labels
labels_string_df <- labels_def %>%
  group_by(barcode) %>%
  summarise(
    # all labels combined
    labels_string  = paste(sort(unique(Canonical2)), collapse = "|"),
    n_labels_total = n_distinct(Canonical2),
    
    # intrinsic labels only
    labels_1_intrinsic = paste(
      sort(unique(Canonical2[str_detect(Category, regex("^1", ignore_case = TRUE))])),
      collapse = "|"),
    n_1_intrinsic = n_distinct(Canonical2[str_detect(Category, regex("^1", ignore_case = TRUE))]),
    
    # extrinsic labels only
    labels_2_extrinsic = paste(
      sort(unique(Canonical2[str_detect(Category, regex("^2", ignore_case = TRUE))])),
      collapse = "|"),
    n_2_extrinsic = n_distinct(Canonical2[str_detect(Category, regex("^2", ignore_case = TRUE))]),
    
    # packaging labels only
    labels_3_packaging = paste(
      sort(unique(Canonical2[str_detect(Category, regex("^3", ignore_case = TRUE))])),
      collapse = "|"),
    n_3_packaging = n_distinct(Canonical2[str_detect(Category, regex("^3", ignore_case = TRUE))]),
    
    .groups = "drop"
  )

# Create data_def
data_def <- category_mapped_filtered %>%
  # Join new label info
  left_join(labels_string_df, by = "barcode") %>%
  # Remove rows with missing or empty labels_string
  filter(!is.na(labels_string) & str_squish(labels_string) != "") %>%
  # Remove invalid barcode
  filter(barcode != "00000000") %>%
  # Drop unneeded columns
  select(-any_of(c("labels_tags", "n_labels")))

# --------------------------- 4. Output  --------------------------------#
write_csv(data_def, "gen/label_mapped_filtered.csv")
write_csv(labels_def, "gen/labels_def.csv")
write_csv(labels_string_df, "gen/labels_string_df.csv")
