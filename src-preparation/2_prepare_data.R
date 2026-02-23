# ---------------------------- 1. Setup ---------------------------------------#
# Setup the environment:
rm(list=ls())
gc()
cat("\014")
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(dplyr, tidyr, purrr, tibble, readr, stringr, here, jsonlite, ggplot2)

# ---------------------------- 2. Input ---------------------------------------#

data <- read_csv("gen/off_selected_fields_filtered.csv")

main_data <- data %>% select(barcode, generic_name_en, labels_tags, nutriscore_2021_tags, nutriscore_2023_tags, nutriscore_score, nova_group, categories_hierarchy)
rm(data); gc()

# --------------------------- 3. Preparation  --------------------------------#
# normalize + split labels
labels_long <- main_data %>%
  transmute(
    barcode,
    label = str_split(labels_tags, "\\s*\\|\\s*")
  ) %>%
  unnest_longer(label, values_to = "label") %>%
  mutate(label = str_squish(str_remove(label, "^[a-z]{2}:"))) %>%
  filter(!is.na(label), label != "") %>%
  distinct(barcode, label)


# counts per product
main_data <- main_data %>%
  left_join(
    labels_long %>%
      count(barcode, name = "n_labels"),
    by = "barcode"
  ) %>%
  mutate(n_labels = replace_na(n_labels, 0))

# frequencies
label_freq <- labels_long %>% count(label, sort = TRUE)

# normalize + split categories to long
categories_long <- main_data %>%
  transmute(
    barcode,
    cat = str_split(categories_hierarchy, "\\s*\\|\\s*")
  ) %>%
  unnest_longer(cat, values_to = "cat") %>%
  mutate(cat = str_squish(str_remove(cat, "^[a-z]{2}:"))) %>%
  filter(!is.na(cat), cat != "") %>%
  distinct(barcode, cat) %>%
  group_by(barcode) %>%
  mutate(level = row_number(), depth = n()) %>%
  ungroup()

# categories to wide
categories_wide <- categories_long %>%
  mutate(col = paste0("cat_level", level)) %>%
  select(barcode, col, cat) %>%
  distinct() %>%
  pivot_wider(names_from = col, values_from = cat)

# deepest category per product
categories_last <- categories_long %>%
  filter(level == depth) %>%
  transmute(barcode, cat_last = cat)

# get frequencies
cat_level1_freq <- categories_long %>% filter(level == 1) %>% count(cat, sort = TRUE)
cat_level2_freq <- categories_long %>% filter(level == 2) %>% count(cat, sort = TRUE)
cat_level3_freq <- categories_long %>% filter(level == 3) %>% count(cat, sort = TRUE)
cat_last_freq   <- categories_last %>% count(cat_last, sort = TRUE)

# join 1-6 and leaf back to main_data 
main_data <- main_data %>%
  left_join(categories_wide %>% select(barcode, cat_level1, cat_level2, cat_level3, cat_level4, cat_level5, cat_level6), 
            by = "barcode") %>%
  left_join(categories_last, by = "barcode")


# counts of barcodes without categories
main_data %>% filter(cat_level1 == "null") %>% summarise(n_barcodes = n_distinct(barcode))

# --------------------------- 4. Output  --------------------------------#
write_csv(main_data, "gen/main_data.csv")
write_csv(label_freq, "gen/label_freq.csv")
write_csv(labels_long, "gen/labels_long.csv")
write_csv(categories_long, "gen/categories_long.csv")
write_csv(categories_wide, "gen/categories_wide.csv")