# ---------------------------- 1. Setup ---------------------------------------#
# Setup the environment:
rm(list=ls())
gc()
cat("\014")
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(dplyr, tidyr, purrr, tibble, readr, stringr, here, jsonlite, ggplot2, widyr, scales, patchwork)

if(!dir.exists("plots")) dir.create("plots")

# ---------------------------- 2. Input ---------------------------------------#

data_def <- read_csv("results/data_def.csv")
label_freq <- read_csv("gen/label_freq.csv")
labels_long <- read_csv("gen/labels_long.csv")

# --------------------------- 3. Statistics  --------------------------------#

glimpse(data_def)
summary(data_def)

# Nutri-Score distributions
nutri <- ggplot(data_def, aes(x = nutriscore, fill = nutriscore)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = scales::comma(..count..)), vjust = -0.3 , size = 16/3) +
  scale_fill_manual(values = c("a"="#8BC34A","b"="#CDDC39","c"="#FFC107","d"="#FF9800","e"="#F44336")) +
  labs(title="Nutri-Score distribution", x="Nutri-Score", y="Count") +
  theme_minimal() + theme(legend.position = "none",
  plot.title = element_text(size = 28), 
  axis.title = element_text(size = 16),
  axis.text  = element_text(size = 16))

# Nova group distribution
nova <- ggplot(data_def, aes(x = factor(nova_group), fill = factor(nova_group))) +
  geom_bar() +
  geom_text(stat = "count", aes(label = scales::comma(..count..)), vjust = -0.3, size = 16/3) +
  scale_fill_manual(values = c("1"="#4CAF50","2"="#8BC34A","3"="#FFC107","4"="#E57373")) +
  labs(title = "NOVA group distribution", x = "NOVA group", y = "Count") +
  theme_minimal() + theme(legend.position = "none",
  plot.title = element_text(size = 28), 
  axis.title = element_text(size = 16),
  axis.text  = element_text(size = 16))

nutri_nova <- nutri+nova

# Histogram of number of labels
labels <- ggplot(data_def %>% filter(n_labels_total <= 25), aes(x = n_labels_total)) +
  geom_bar(fill = "steelblue") +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Number of labels per product",
       x = "Number of labels", y = "Number of products") +
  theme_minimal() + 
  theme(
    plot.title = element_text(size = 28),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 16)
  )

# Nutri-score distribution within NOVA groups
dist_nutri_nova <- data_def %>%
  count(nova_group, nutriscore) %>%
  group_by(nova_group) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(y = factor(nova_group), x = prop, fill = nutriscore)) +
  geom_col(position = position_fill(reverse = TRUE)) +
  geom_text(aes(label = scales::percent(prop, accuracy = 1)),
            position = position_fill(vjust = 0.5, reverse = TRUE),
            color = "white", size = 16/3) +
  scale_fill_manual(values = c("a"="#8BC34A","b"="#CDDC39","c"="#FFC107","d"="#FF9800","e"="#F44336")) +
  labs(title = "Nutri-Score distribution within NOVA groups",
       y = "NOVA group", x = "Share within NOVA group",
       fill = "Nutri-Score") +
  scale_x_continuous(labels = scales::percent) + theme_minimal() +
  theme(
    plot.title = element_text(size = 28),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 16),
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 16)
  )

# Map to broader categories
category_lookup <- tribble(
  ~mapped_category, ~main_mapped_category,
  "1.1 Artificially sweetened beverages", "1. Beverages",
  "1.2 Fruit juices and nectars", "1. Beverages",
  "1.3 Sweetened beverages", "1. Beverages",
  "1.4 Unsweetened beverages", "1. Beverages",
  "1.5 Alcoholic beverages", "1. Beverages",
  "1.6 Plant-based milk substitutes", "1. Beverages",
  "2.1 Fruits", "2. Fruits & Vegetables",
  "2.2 Vegetables", "2. Fruits & Vegetables",
  "2.3 Legumes", "2. Fruits & Vegetables",
  "2.4 Soups", "2. Fruits & Vegetables",
  "2.5 Potatoes", "2. Fruits & Vegetables",
  "3.1 Bread", "3. Cereals & Starches",
  "3.2 Cereals", "3. Cereals & Starches",
  "3.3 Breakfast cereals", "3. Cereals & Starches",
  "3.4 Composite starchy foods", "3. Cereals & Starches",
  "4.1 Milk and yogurt", "4. Dairy",
  "4.2 Cheese", "4. Dairy",
  "4.3 Dairy desserts", "4. Dairy",
  "4.4 Ice cream", "4. Dairy",
  "5.1 Processed meat", "5. Meat, Fish & Eggs",
  "5.2 Fish and seafood", "5. Meat, Fish & Eggs",
  "5.3 Eggs", "5. Meat, Fish & Eggs",
  "5.4 Meat", "5. Meat, Fish & Eggs",
  "6.1 Fats & oils", "6. Fats & Sauces",
  "6.2 Dressings and sauces", "6. Fats & Sauces",
  "7.1 One-dish meals", "7. Composite & Prepared Meals",
  "7.2 Pizza, pies and quiche", "7. Composite & Prepared Meals",
  "7.3 Sandwiches & burgers", "7. Composite & Prepared Meals",
  "8.1 Appetisers", "8. Snacks & Appetisers",
  "8.2 Nuts and Seeds", "8. Snacks & Appetisers",
  "8.3 Salty spreads and fatty snacks", "8. Snacks & Appetisers",
  "9.1 Biscuits and cakes", "9. Sweet Products & Desserts",
  "9.2 Pastries", "9. Sweet Products & Desserts",
  "9.3 Chocolate products", "9. Sweet Products & Desserts",
  "9.4 Sweets", "9. Sweet Products & Desserts",
  "9.5 Sugary desserts", "9. Sweet Products & Desserts",
  "Other products", "Other products"
)

# Create statistics per main category
categories_mapping_stats <- data_def %>%
  group_by(mapped_category) %>%
  summarise(
    count = n(),
    nutriscore_a = sum(nutriscore == "a", na.rm = TRUE),
    nutriscore_b = sum(nutriscore == "b", na.rm = TRUE),
    nutriscore_c = sum(nutriscore == "c", na.rm = TRUE),
    nutriscore_d = sum(nutriscore == "d", na.rm = TRUE),
    nutriscore_e = sum(nutriscore == "e", na.rm = TRUE),
    nova_group_1 = sum(nova_group == 1, na.rm = TRUE),
    nova_group_2 = sum(nova_group == 2, na.rm = TRUE),
    nova_group_3 = sum(nova_group == 3, na.rm = TRUE),
    nova_group_4 = sum(nova_group == 4, na.rm = TRUE)
  ) %>%
  mutate(
    percent = count / sum(count) * 100
  ) %>%
  arrange(desc(count))

categories_mapping_stats <- categories_mapping_stats %>%
  left_join(category_lookup, by = "mapped_category") %>%
  relocate(main_mapped_category, .after = mapped_category)

categories_main_stats <- data_def %>%
  left_join(category_lookup, by = "mapped_category") %>%
  group_by(main_mapped_category) %>%
  summarise(
    count = n(),
    nutriscore_a = sum(nutriscore == "a", na.rm = TRUE),
    nutriscore_b = sum(nutriscore == "b", na.rm = TRUE),
    nutriscore_c = sum(nutriscore == "c", na.rm = TRUE),
    nutriscore_d = sum(nutriscore == "d", na.rm = TRUE),
    nutriscore_e = sum(nutriscore == "e", na.rm = TRUE),
    nova_group_1 = sum(nova_group == 1, na.rm = TRUE),
    nova_group_2 = sum(nova_group == 2, na.rm = TRUE),
    nova_group_3 = sum(nova_group == 3, na.rm = TRUE),
    nova_group_4 = sum(nova_group == 4, na.rm = TRUE)
  ) %>%
  mutate(percent = count / sum(count) * 100) %>%
  arrange(desc(count))

combined_main_long <- categories_main_stats %>%
  select(main_mapped_category, starts_with("nutriscore_"), starts_with("nova_group_")) %>%
  pivot_longer(
    cols = -main_mapped_category,
    names_to = "variable",
    values_to = "count"
  ) %>%
  mutate(
    type  = ifelse(grepl("^nutriscore_", variable), "Nutri-Score", "NOVA Group"),
    label = toupper(gsub("^nutriscore_|^nova_group_", "", variable)),
    label = factor(label, levels = c("1","2","3","4","A","B","C","D","E"))
  )

# Nutri-score distribution within NOVA groups per category
categories <- ggplot(combined_main_long %>% group_by(main_mapped_category,type) %>%
         mutate(prop=count/sum(count),
                label_text=ifelse(prop>0.05,paste0(label," (",scales::percent(prop,accuracy=1),")"),"")),
       aes(x=factor(main_mapped_category,levels=sort(unique(main_mapped_category),decreasing=TRUE)),
           y=prop,fill=label))+
  geom_col(position=position_fill(reverse=TRUE))+
  geom_text(aes(label=label_text),
            position=position_fill(vjust=0.5,reverse=TRUE),color="white",size=16/3)+
  facet_wrap(~type,ncol=1,scales="free_y")+
  scale_y_continuous(labels=scales::percent)+
  scale_fill_manual(values=c("1"="#4CAF50","2"="#8BC34A","3"="#FFC107","4"="#E57373",
                             "A"="#8BC34A","B"="#CDDC39","C"="#FFC107","D"="#FF9800","E"="#F44336"),
                    breaks=c("1","2","3","4","A","B","C","D","E"),name="Label")+
  labs(title="Category balance (main level): Nutri-Score and NOVA Group distributions",
       x="Main Category",y="Share within Category")+
  coord_flip()+theme_minimal() +
  theme(
    plot.title  = element_text(size = 28, hjust=0.5),
    axis.title  = element_text(size = 16),
    axis.text   = element_text(size = 16),
    strip.text  = element_text(size = 20, face = "bold"), 
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 16)
  )

# --------------------------- 4. Output  --------------------------------#

ggsave("plots/nutriscore_nova.png", plot = nutri_nova, width = 14, height = 7, dpi = 300)
ggsave("plots/labels.png", plot = labels, width = 14, height = 7, dpi = 300, bg = "white")
ggsave("plots/dist_nutri_nova.png", plot = dist_nutri_nova, width = 14, height = 7, dpi = 300, bg = "white")
ggsave("plots/categories.png", plot = categories, width = 16, height = 10, dpi = 300, bg = "white")
