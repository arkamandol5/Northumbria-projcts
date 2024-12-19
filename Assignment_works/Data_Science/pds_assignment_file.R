# Loading the Dataset

# Loading necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)

# Reading the CSV file into a dataframe
data <- read_csv("World University Rankings (1).csv")




# DATA CLEANING PART

data <- data %>% 
  mutate(scores_teaching = as.character(scores_teaching),
         scores_research = as.character(scores_research),
         scores_citations = as.character(scores_citations),
         scores_industry_income = as.character(scores_industry_income),
         scores_international_outlook = as.character(scores_international_outlook),
         overall_score = as.character(overall_score)) %>%
  mutate(across(c(scores_teaching, scores_research, scores_citations, 
                  scores_industry_income, scores_international_outlook,overall_score), 
                ~na_if(., "n/a")))

missing <- is.na(data$scores_teaching)
missing_sum <- sum(missing == TRUE)
missingness <- (missing_sum/2673)*100

# Checking for constant columns
constant_columns <- sapply(data, function(x) length(unique(x)) == 1)
# Print constant columns
print(names(data)[constant_columns])
# If there are constant columns, you might consider removing them
data <- data %>% select(-one_of(names(data)[constant_columns]))



library(naniar)
vis_miss(data)
mcar_test(data)

# remove rows from your dataframe in R that have NA in the scores_teaching column
data <- data %>% 
  filter(!is.na(scores_teaching))


library(dplyr)
library(stringr)

# Function to calculate the average if the value is a range in overall score column
calculate_average <- function(x) {
  if (str_detect(x, "–")) {
    nums <- as.numeric(str_split(x, "–", simplify = TRUE))
    return(mean(nums))
  } else {
    return(as.numeric(x))
  }
}

# Applying the function to overall_score column to put a average value to range values
data <- data %>%
  mutate(overall_score = sapply(overall_score, calculate_average))

# taking backup of the data
dataset_backup <- data
#removing any duplicate values if present
data <- data %>% distinct()


# Convert 'stats_number_students' to numeric by removing commas
data$stats_number_students <- as.numeric(gsub(",", "", data$stats_number_students))

# Convert percentages in 'stats_pc_intl_students' to numeric
data$stats_pc_intl_students <- as.numeric(sub("%", "", data$stats_pc_intl_students)) / 100

# Handling 'stats_female_male_ratio' and convert to numeric ratio
data <- data %>%
  mutate(female_ratio = as.numeric(gsub(":.*", "", stats_female_male_ratio)),
         male_ratio = as.numeric(gsub(".*:", "", stats_female_male_ratio))) %>%
  select(-stats_female_male_ratio)


data$female_ratio <- ifelse(is.na(data$female_ratio), 
                                       mean(data$female_ratio, na.rm = TRUE), 
                                       data$female_ratio)
data$male_ratio <- ifelse(is.na(data$male_ratio), 
                            mean(data$male_ratio, na.rm = TRUE), 
                            data$male_ratio)


vis_miss(data)



# eda part.

# Ensure numeric data types for plotting
data$scores_teaching <- as.numeric(data$scores_teaching)
data$scores_research <- as.numeric(data$scores_research)
data$scores_citations <- as.numeric(data$scores_citations)
data$scores_international_outlook <- as.numeric(data$scores_international_outlook)
data$overall_score <- as.numeric(data$overall_score)

# Handling NAs in these columns if they exist
data <- na.omit(data, cols = c('scores_teaching', 'scores_research', 
                               'scores_citations', 'scores_international_outlook', 'overall_score'))

# Plotting the correlation matrix
library(corrplot)
corr_matrix <- cor(data[, sapply(data, is.numeric)])
corrplot(corr_matrix, method = "color")

# Plotting scatter plots of key features against the overall score
feature_names <- c('scores_teaching', 'scores_research', 'scores_citations', 'scores_international_outlook')
for (feature in feature_names) {
  ggplot(data, aes(x = !!sym(feature), y = overall_score)) +
    geom_point() +
    ggtitle(paste("Scatter Plot of", feature, "vs Overall Score"))
}


library(ggplot2)

# Loop through each numerical column (excluding 'overall_score') and create a scatter plot
numeric_columns <- sapply(data, is.numeric)
features <- names(numeric_columns)[numeric_columns]
features <- setdiff(features, "overall_score") # exclude 'overall_score' from features

# Generate scatter plots
pdf("scatter_plots.pdf") # Save plots to a PDF file
for (feature in features) {
  p <- ggplot(data, aes_string(x = feature, y = "overall_score")) +
    geom_point() +
    ggtitle(paste("Scatter Plot of", feature, "vs Overall Score")) +
    xlab(feature) +
    ylab("Overall Score")
  print(p) # Print the plot
}
dev.off() # Close the PDF device


keep_numerical <- function(data) {
  library(dplyr)
  data %>%
    select_if(is.numeric)
}


new_data <- keep_numerical(data)
print(new_data)  

install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
chart.Correlation(new_data, histogram = TRUE)


library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(rpart)
# Model Building and Evaluation

# Splitting the data into training and testing sets
set.seed(42) # for reproducibility
trainIndex <- createDataPartition(new_data$overall_score, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- new_data[trainIndex, ]
data_test <- new_data[-trainIndex, ]

# Defining models to be tested
model_list <- list(linear_reg = lm(overall_score ~ ., data = data_train),
                   decision_tree = rpart(overall_score ~ ., data = data_train, method = "anova"),
                   random_forest = randomForest(overall_score ~ ., data = data_train))

# Evaluating models
results <- list()
for (model_name in names(model_list)) {
  model <- model_list[[model_name]]
  predictions <- predict(model, newdata = data_test)
  mse <- mean((predictions - data_test$overall_score)^2)
  r2 <- cor(predictions, data_test$overall_score)^2
  results[[model_name]] <- list(MSE = mse, R2 = r2)
}

cor(data_test$overall_score, predictions, method = "pearson")-

# Feature importance from the Random Forest model
importance <- randomForest(overall_score ~ ., data = data_train)$importance
importance_df <- as.data.frame(importance)

# Plotting feature importances
ggplot(importance_df, aes(x = reorder(row.names(importance_df), IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Features") +
  ylab("Importance") +
  ggtitle("Feature Importances in Random Forest Model")

