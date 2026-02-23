# ---------------------------- 1. Setup ---------------------------------------#
# Setup the environment:
rm(list=ls())
gc()
cat("\014")
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(dplyr, tidyr, purrr, tibble, readr, stringr, here, jsonlite, ggplot2, widyr, scales,tidyverse, reshape2, Matrix, fastcluster, coop, dendextend, RColorBrewer, igraph, knitr)

# ---------------------------- 2. Input ---------------------------------------#

main_data <- read_csv("gen/main_data.csv")
categories_wide <- read_csv("gen/categories_wide.csv")
categories_long <- read_csv("gen/categories_long.csv")

# ---------------------------- 2. Mapping -------------------------------------#

# Get frequencies of ALL categories
cat_freq <- categories_long %>%
  count(cat, name = "n_cat", sort = TRUE)

# Get frequencies PER level PER category
cat_freq_levels <- categories_long %>%
  # Count occurrences per category per level
  count(cat, level, name = "n_level") %>%
  # Spread levels into wide format (n_level1 ... n_level30)
  pivot_wider(
    names_from = level,
    values_from = n_level,
    names_prefix = "n_level",
    values_fill = 0
  ) %>%
  # Ensure numeric ordering of columns (1,2,3,...,30)
  select(cat, starts_with("n_level")) %>%
  {
    # reorder the n_level columns numerically
    lvl_cols <- select(., starts_with("n_level")) %>% names()
    lvl_cols_ordered <- lvl_cols[order(as.numeric(str_extract(lvl_cols, "\\d+")))]
    select(., cat, all_of(lvl_cols_ordered))
  } %>%
  # Add total frequency and reorder columns
  mutate(n_cat = rowSums(across(starts_with("n_level")))) %>%
  relocate(n_cat, .after = cat) %>%
  arrange(desc(n_cat))

############ Mapping by hand

# PRIORITY: smaller = wins if multiple matches
category_mapping_rules <- tribble(
  ~target,                          ~tokens,                                                                                  ~priority,
  # Beverages
  "1.1 Artificially sweetened beverages", c("artificially-sweetened-beverages","zero-sugar-drinks","diet-sodas?"),                1,
  "1.2 Fruit juices and nectars",        c("juices-and-nectars","fruit-juices?","nectars?", "^smoothies$"),                                      2,
  "1.3 Sweetened beverages",             c("sweetened-beverages","sodas?|soft-drinks?","energy-drinks?", "^iced-coffees$", "^aloe-vera-drinks$"), 3,
  "1.4 Unsweetened beverages",           c("unsweetened-beverages","waters?","coffee","teas?", "^ground-coffees$", "^arabica-coffees$", "^instant-coffees$", "^decaffeinated-coffees$", "^coffees$"),    4,
  "1.5 Alcoholic beverages",             c("(?<!non-)alcoholic-beverages\\b", "beers?", "wines?", "spirits"),                     2,
  "1.6 Plant-based milk substitutes",    c("plant-based-milk-alternatives","plant-milks?","soy-drinks?","oat-drinks?","almond", "^plant-based-creams$"),  2,
  
  # Fruits & Vegetables
  "2.1 Fruits",                          c("fruits","dried-fruits?"),                                                              3,
  "2.2 Vegetables",                      c("vegetables", "hearts-of-palm", "^sprouts$","^olives$", "^pickles$", "^pumpkin-and-squash-plant-products$"),           3,
  "2.3 Legumes",                         c("legumes"),                                                                             2,
  "2.4 Soups",                           c("soups", "broths"),                                                                               2,
  "2.5 Potatoes",                        c("potatoes","sweet-potatoes"),                                                           2,
  
  # Cereals & Starches
  "3.1 Bread",                           c("breads?"),                                                                             2,
  "3.2 Cereals",                         c("cereal-pastas?", "cereal-grains?", "cereal-flours?", "rices?", "pasta", "^flours$", "^coquillettes$", "^noodles$", "^dry-pastas$", "^fusilli$", "^konjac$", "^lasagna-sheets$", "^linguine$", "^penne$", "^rigatoni$", "^spaghetti$", "^spatzle$", "^tagliatelle$"), 3,
  "3.3 Breakfast cereals",               c("breakfast-cereals", "\\bcereal[-\\s]?bars?\\b", "^Snacks, Sweet snacks, Bars, Cereal bars$", "^energy-bars$", "^protein-bars$"),            1,
  "3.4 Composite starchy foods",         c("composite-starchy-foods"),                                                             4,  # optional catch
  
  # Dairy & Alternatives
  "4.1 Milk and yogurt",                 c("milks?","yogurts?","fermented-milks?"),                                                2,
  "4.2 Cheese",                          c("cheeses?"),                                                                            1,
  "4.3 Dairy desserts",                  c("dairy-desserts"),                                                                      2,
  "4.4 Ice cream",                       c("ice-creams?-and-sorbets?","ice-creams?","sorbets?"),                                   1,
  
  # Meat, Fish & Eggs
  "5.1 Processed meat",                  c("prepared-meats?","processed-meats?","sausages?","bacon","ham","salami","cold-cuts?"), 1,
  "5.2 Fish and seafood",                c("seafood","fish-and-seafood"),                                                          2,
  "5.3 Eggs",                            c("eggs?","omelettes?","egg-based-products?"),                                           2,
  "5.4 Meat",                            c("meats?","poultries?","beef","pork","chicken","turkey"),                                4,
  
  # Fats & Sauces
  "6.1 Fats & oils",                     c("butters?","margarines?","vegetable-fats","spreadable-fats","oils?"),                  2,
  "6.2 Dressings and sauces",            c("sauces","condiments","mayonnaise","ketchup","dressings?"),                            3,
  
  # Composite & Prepared Meals
  "7.1 One-dish meals",                  c("microwave-meals", "ready-made-meals", "frozen-ready-made-meals", "combination-meals", "^canned-meals$", "^prepared-salads$", "^plant-based-meals$", "^refrigerated-meals$", "^cod-brandade$", "^filled-buckwheat-crepes$", "^quenelles$", "^sushi-and-maki$", "^meals-with-fish$", "^meals-with-shellfish$", "^gratins$", "^prepared-couscous$", "^tabbouleh$", "^nems$", "^samosas$", "^spring-rolls$"), 2,
  "7.2 Pizza, pies and quiche",          c("pizzas-pies-and-quiches?","pizzas?","quiches?","savory-pies?"),                       1,
  "7.3 Sandwiches & burgers",            c("sandwiches","hamburgers?","burgers?","panini"),                                       2,
  
  # Snacks & Appetisers
  "8.1 Appetisers",                      c("crackers-appetizers","crisps","chips","crackers", "^popcorn$", "^Snacks, Popcorn$"),      2,
  "8.2 Nuts and Seeds",                  c("nuts", "nut-and-seed-mixes", "^sunflower-seeds$", "^sesame$", "^seeds$"),                   1,
  "8.3 Salty spreads and fatty snacks",  c("salted-spreads","chestnut-spreads","savory-spreads?","hummus","tapenade", "^plant-based-pates$", "^terrines$"),   3,
  
  # Sweet Products & Desserts
  "9.1 Biscuits and cakes",              c("biscuits-and-cakes","cookies","cakes", "pancakes", "^filled-crepes$", "^friands$"),                                   2,
  "9.2 Pastries",                        c("pastries","croissants","brioches","donuts?"),                                         2,
  "9.3 Chocolate products",              c("cocoa-and-its-products","chocolate-bars?","chocolate-spreads?", "^Snacks, Sweet snacks, Cocoa and its products, Confectioneries, Chocolate candies$"),        2,
  "9.4 Sweets",                          c("gummi-candies", "gums", "honeys?", "jams?", "sweets", "^Snacks, Sweet snacks, Confectioneries$", "^candies$", "^caramels$", "^Gummies$", "^marzipan$", "^nougats$", "^fruit-jellies$", "^marmalades$", "^syrups$", "^sugars$", "^confectioneries$"), 3,
  "9.5 Sugary desserts",                 c("non-dairy-desserts","puddings?","mousse"),                                            3,
  
  # Other
  "Other products",                  c("other"),                                                                               9
)

rules <- category_mapping_rules %>%
  mutate(
    # one big OR-regex per row; match whole tokens anywhere in the OFF path
    pattern = map_chr(tokens, ~ str_c("\\b(", str_c(.x, collapse="|"), ")\\b", collapse="")),
    tokens = NULL
  )

# 1) find all matches (barcode x rule)
rule_hits <- rules %>%
  dplyr::mutate(rule_id = dplyr::row_number()) %>%
  dplyr::select(rule_id, target, priority, pattern) %>%
  tidyr::crossing(categories_long %>% dplyr::select(barcode, cat, level)) %>%
  dplyr::filter(stringr::str_detect(cat, stringr::regex(pattern, ignore_case = TRUE))) %>%
  dplyr::select(barcode, level, target, priority)

# 2) resolve conflicts: priority asc, then deepest level desc
harmonized_map <- rule_hits %>%
  arrange(barcode, priority, desc(level)) %>%
  group_by(barcode) %>%
  slice(1) %>%
  ungroup() %>%
  rename(harmonized_category = target)

# 3) fallback when no rule matched: use cat_level1 or "Other products"
fallback <- main_data %>%
  select(barcode, cat_level1) %>%
  mutate(harmonized_category = coalesce(cat_level1, "Other products")) %>%
  select(barcode, harmonized_category)

# prefer matched label; else fallback
harmonized_final <- fallback %>%
  left_join(harmonized_map, by = "barcode", suffix = c("_fallback","")) %>%
  transmute(
    barcode,
    harmonized_category = coalesce(harmonized_category, harmonized_category_fallback)
  )

# Check fallbacks
rule_targets <- unique(rules$target)

# Map back source
harmonized_final <- harmonized_final %>%
  mutate(
    source = case_when(
      !is.na(harmonized_category) & harmonized_category %in% rule_targets ~ "rule",
      TRUE ~ "fallback"
    )
  )

harmonized_final %>% count(source) %>% mutate(pct = n/sum(n)*100)

uncategorized <- harmonized_final %>% 
  filter(source == "fallback") %>%
  count(harmonized_category, sort = TRUE) %>%
  mutate(pct = n/sum(n)*100)

cat_harmonized_freq <- harmonized_final %>%
  filter(source == "rule") %>%
  count(harmonized_category, sort = TRUE) %>%
  mutate(pct = n / sum(n) * 100)

# Add harmonized category and source to categories_wide
categories_wide <- categories_wide %>%
  left_join(
    harmonized_final %>% select(barcode, harmonized_category, source),
    by = "barcode"
  ) %>%
  relocate(barcode, harmonized_category, source, .before = everything())

categories_wide <- categories_wide %>% 
  left_join(
    main_data %>% select(barcode, generic_name_en),
    by = "barcode"
  ) %>%
  relocate(barcode, generic_name_en, .before = everything())

# Add final mapped category to harmonized_final 
harmonized_final <- harmonized_final %>%
  dplyr::mutate(
    mapped_category = dplyr::if_else(source == "rule", harmonized_category, "Other products")
  )

# Create final_data with new mapped category
final_data <- main_data %>% 
  select(
    barcode,
    product_name = generic_name_en,
    nutriscore = nutriscore_2023_tags,
    nova_group,
    labels_tags,
    n_labels
  ) %>%
  # bring in the mapped_category from harmonized_final via barcode
  left_join(
    harmonized_final %>% select(barcode, mapped_category),
    by = "barcode"
  ) %>%
  # if a barcode didn't get a mapping, call it "Others"
  mutate(mapped_category = replace_na(mapped_category, "Other products")) %>%
  select(
    barcode, product_name, nutriscore, nova_group, mapped_category, labels_tags, n_labels
  )

final_data_filtered <- final_data %>%
  filter(mapped_category != "Other products")


# --------------------------- 4. Output  --------------------------------#
write_csv(rule_hits, "gen/rule_hits.csv")
write_csv(cat_harmonized_freq, "gen/cat_harmonized_freq.csv")
write_csv(final_data, "gen/category_mapped_data.csv")
write_csv(final_data_filtered, "gen/category_mapped_filtered.csv")
