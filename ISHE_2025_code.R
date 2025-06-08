# Load necessary libraries
library(brms)
library(tidyverse)
library(bayestestR)
library(dplyr)
library(psych)

# Load the dataset (select the file manually)
data <- read.csv(file.choose())

# View the structure of the dataset
glimpse(data)

# Select and rename the relevant variables for clarity
data <- data %>%
  select(q0002, q0003, q0004, q0005, Apego_ansiedade,
         Apego_evitativo, Adaptacao, Autonomia, Afetivo) %>%
  rename(
    gender = q0002,
    age = q0003,
    education = q0004,
    residence = q0005,
    anxiety_attach = Apego_ansiedade,
    avoidance_attach = Apego_evitativo,
    adaptation = Adaptacao,
    autonomy = Autonomia,
    affective = Afetivo
  )

# Check if the variables names were changed
colnames(data)
# Create a binary gender variable: female = 1, male = 0
data <- data %>%
  mutate(female = case_when(
    gender == 1 ~ 1,
    gender == 2 ~ 0,
    TRUE ~ NA_real_
  ))

# Recode education variable
data <- data %>%
  mutate(education = case_when(
    education == 1 ~ "Incomplete elementary school",
    education == 2 ~ "Complete elementary school",
    education == 3 ~ "Incomplete high school",
    education == 4 ~ "Complete high school",
    education == 5 ~ "Incomplete undergraduate",
    education == 6 ~ "Complete undergraduate",
    education == 7 ~ "Incomplete postgraduate",
    education == 8 ~ "Complete postgraduate",
    TRUE ~ NA_character_
  ))
# Recode state of residence into Brazil's five regions
data <- data %>%
  mutate(residence = case_when(
    residence %in% c(1, 3, 4, 14, 22, 23, 27) ~ "North",              
    residence %in% c(2, 5, 6, 10, 15, 17, 18, 20, 26) ~ "Northeast",
    residence %in% c(9, 11, 12, 7) ~ "Central-West",                
    residence %in% c(8, 13, 19, 25) ~ "Southeast",                   
    residence %in% c(16, 21, 24) ~ "South",
    residence == 28 ~ "Not living in Brazil",
    TRUE ~ NA_character_
  ))
# --- Descriptive Statistics ---

# Get mean and standard deviation for age using the psych package
describe(data$age)[, c("mean", "sd")]

# Frequency and percentage tables for categorical variables
table(data$gender)
round(prop.table(table(data$gender)), 4) * 100

table(data$education)
round(prop.table(table(data$education)), 4) * 100

table(data$residence)
round(prop.table(table(data$residence)), 4) * 100

# --- Data Preparation for Modeling ---

# Remove rows with missing values in the main variables
data <- data %>%
  drop_na(anxiety_attach, avoidance_attach, adaptation, autonomy, affective)

# Convert variables to numeric (if not already)
data$anxiety_attach <- as.numeric(data$anxiety_attach)
data$avoidance_attach <- as.numeric(data$avoidance_attach)
data$adaptation <- as.numeric(data$adaptation)
data$autonomy <- as.numeric(data$autonomy)
data$affective <- as.numeric(data$affective)

# Standardize the variables (z-scores)
data_z <- data
data_z$adaptation <- scale(data$adaptation)
data_z$autonomy <- scale(data$autonomy)
data_z$affective <- scale(data$affective)
data_z$anxiety_attach <- scale(data$anxiety_attach)
data_z$avoidance_attach <- scale(data$avoidance_attach)

# --- Bayesian Model: Anxious Attachment ---

# Fit the full model
anxiety_standardized <- brm(
  anxiety_attach ~ adaptation + autonomy + affective,
  data = data_z,
  family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

# Inspect model results
summary(anxiety_standardized)
posterior_summary(anxiety_standardized)
pp_check(anxiety_standardized)
bayes_R2(anxiety_standardized)

# --- Bayesian Model: Avoidant Attachment ---

# Fit the full model
avoidance_standardized <- brm(
  avoidance_attach ~ adaptation + autonomy + affective,
  data = data_z,
  family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

# Inspect model results
summary(avoidance_standardized)
posterior_summary(avoidance_standardized)
pp_check(avoidance_standardized)
bayes_R2(avoidance_standardized)

# --- Bayes Factors: Anxious Attachment ---

# Compare full model to reduced models (removing one predictor at a time)
model_anxiety <- anxiety_standardized

anxiety_without_adaptation <- brm(
  anxiety_attach ~ autonomy + affective,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

anxiety_without_autonomy <- brm(
  anxiety_attach ~ adaptation + affective,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

anxiety_without_affective <- brm(
  anxiety_attach ~ adaptation + autonomy,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

# Compute Bayes factors for each predictor
bf_anxiety_adaptation <- bayes_factor(model_anxiety, anxiety_without_adaptation)
bf_anxiety_autonomy <- bayes_factor(model_anxiety, anxiety_without_autonomy)
bf_anxiety_affective <- bayes_factor(model_anxiety, anxiety_without_affective)

# Print Bayes factors
print(bf_anxiety_adaptation)
print(bf_anxiety_autonomy)
print(bf_anxiety_affective)

# --- Bayes Factors: Avoidant Attachment ---

# Compare full model to reduced models (removing one predictor at a time)
model_avoidance <- avoidance_standardized
  
avoidance_without_adaptation <- brm(
  avoidance_attach ~ autonomy + affective,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

avoidance_without_autonomy <- brm(
  avoidance_attach ~ adaptation + affective,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

avoidance_without_affective <- brm(
  avoidance_attach ~ adaptation + autonomy,
  data = data_z, family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = "b"),
    prior(normal(0, 1), class = "Intercept"),
    prior(normal(0, 1), class = "sigma")
  ),
  chains = 4, iter = 4000, warmup = 1000, seed = 123
)

# Compute Bayes factors for each predictor
bf_avoidance_adaptation <- bayes_factor(model_avoidance, avoidance_without_adaptation)
bf_avoidance_autonomy <- bayes_factor(model_avoidance, avoidance_without_autonomy)
bf_avoidance_affective <- bayes_factor(model_avoidance, avoidance_without_affective)

# Print Bayes factors
print(bf_avoidance_adaptation)
print(bf_avoidance_autonomy)
print(bf_avoidance_affective)
