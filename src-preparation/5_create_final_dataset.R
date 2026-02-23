# ---------------------------- 1. Setup ---------------------------------------#
# Setup the environment:
rm(list=ls())
gc()
cat("\014")
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(readxl, dplyr, tidyr, purrr, tibble, readr, stringr, here, jsonlite, ggplot2, widyr, scales,tidyverse, reshape2, Matrix, fastcluster, coop, dendextend, RColorBrewer, igraph, knitr)

# ---------------------------- 2. Input ---------------------------------------#

data_def <- read_csv("gen/label_mapped_filtered.csv")
data <- read_csv("gen/off_selected_fields_filtered.csv")

if (!dir.exists("results")) dir.create("results")
# ------------------------- 3. Transformation  --------------------------------#

data_controls <- data %>%
  select(barcode, brands, countries_hierarchy, additives_n, ingredients_n) %>%
  mutate(
    countries_hierarchy = countries_hierarchy %>%
      # remove "en:" and similar prefixes
      str_remove_all("([a-z]{2}:)") %>%
      # trim whitespace and replace separators with "|"
      str_squish() %>%
      str_replace_all("\\s*\\|\\s*", "|")
  ) %>%
  rename(
    countries     = countries_hierarchy,
    n_additives   = additives_n,
    n_ingredients = ingredients_n
  )

data_def <- data_def %>%
  left_join(data_controls, by = "barcode")

# function: union pipe-strings -> single pipe-string
combine_labels <- function(...) {
  strings <- c(...)
  toks <- unlist(str_split(strings[!is.na(strings) & strings != ""], "\\|"))
  toks <- sort(unique(str_squish(toks)))
  if (length(toks) == 0) NA_character_ else paste(toks, collapse = "|")
}

# function: count tokens in a pipe-string
count_labels <- function(s) {
  if (is.na(s) || s == "") 0L else length(str_split(s, "\\|")[[1]])
}

# Category 1 only
data_def_cat1 <- data_def %>%
  rowwise() %>%
  mutate(
    labels_string  = labels_1_intrinsic,                 
    n_labels_total = count_labels(labels_string)
  ) %>%
  ungroup() %>%
  filter(!is.na(labels_string) & str_squish(labels_string) != "") %>%
  select(-labels_2_extrinsic, -labels_3_packaging)       

# Categories 1 + 2 
data_def_cat12 <- data_def %>%
  rowwise() %>%
  mutate(
    labels_string  = combine_labels(labels_1_intrinsic, labels_2_extrinsic),
    n_labels_total = count_labels(labels_string)
  ) %>%
  ungroup() %>%
  filter(!is.na(labels_string) & str_squish(labels_string) != "")

# Categories 1 + 2 + 3 (all)
data_def_cat123 <- data_def %>%
  rowwise() %>%
  mutate(
    labels_string  = combine_labels(labels_1_intrinsic, labels_2_extrinsic, labels_3_packaging),
    n_labels_total = count_labels(labels_string)
  ) %>%
  ungroup() %>%
  filter(!is.na(labels_string) & str_squish(labels_string) != "")

# Starting from packaging, Category 3 only
data_def_cat3  <- data_def %>%
  rowwise() %>%
  mutate(labels_string = labels_3_packaging,
         n_labels_total = count_labels(labels_string)) %>%
  ungroup() %>%
  filter(!is.na(labels_string) & str_squish(labels_string) != "")

# Categories 2 + 3 
data_def_cat23 <- data_def %>%
  rowwise() %>%
  mutate(labels_string = combine_labels(labels_2_extrinsic, labels_3_packaging),
         n_labels_total = count_labels(labels_string)) %>%
  ungroup() %>%
  filter(!is.na(labels_string) & str_squish(labels_string) != "")

# --------------------------- 4. Output  --------------------------------#
write_csv(data_def, "results/data_def.csv", na = "")
write_csv(data_def_cat1,   "gen/data_def_cat1.csv",   na = "")
write_csv(data_def_cat12,  "gen/data_def_cat12.csv",  na = "")
write_csv(data_def_cat123, "gen/data_def_cat123.csv", na = "")
write_csv(data_def_cat3,   "gen/data_def_cat3.csv",   na = "")
write_csv(data_def_cat23,  "gen/data_def_cat23.csv",  na = "")
