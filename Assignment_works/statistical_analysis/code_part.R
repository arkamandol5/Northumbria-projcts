setwd("/Users/arkamandol/DataspellProjects/ml_statistical/statistical/stats_assign")

# part 3
# Load data from a text file
cystic_fibrosis_data <- read.table("DataQ3.txt", header = TRUE, sep = "", na.strings = "NA", dec = ".", strip.white = TRUE)


# Check the first few rows to confirm data is loaded correctly
head(cystic_fibrosis_data)

# Basic summary statistics for each variable
summary(cystic_fibrosis_data)

summary(cystic_fibrosis_data$age)
summary(cystic_fibrosis_data$sex)
summary(cystic_fibrosis_data$height)

#Given the summaries you've provided, let's comment on them:

#- **Age**: The age of participants ranges from 7 to 23 years, with a median age of 14 years. This suggests that the participants are primarily adolescents.

#- **Sex**: The binary representation of sex (assuming 0 for females and 1 for males) shows a fairly balanced distribution, though without the exact counts, the mean suggests a slightly higher proportion of males.

#- **Height**: Heights range from 109cm to 180cm, indicating a wide range of developmental stages consistent with the age range.

#- **Weight**: Weights range from 12.9kg to 73.8kg, further indicating a broad spectrum of development among participants.

#- **BMP, FEV1, RV, FRC, TLC, PEMAX**: These variables are likely specific medical or physiological measurements relevant to the study of cystic fibrosis. Their values span a wide range, which could indicate varying degrees of disease severity or progression among the participants.
#- **BMP** and **FEV1** seem to pertain to respiratory function, with their minimum and maximum values suggesting a wide variability in lung function.
#- **RV (Residual Volume)**, **FRC (Functional Residual Capacity)**, and **TLC (Total Lung Capacity)** are measures that provide insights into lung volume and capacity, crucial for assessing cystic fibrosiss impact.
#- **PEMAX** might relate to the maximum expiratory pressure, another respiratory measure.


# part 4
# Load the ggplot2 library for plotting
library(ggplot2)

# Scatterplot of age vs fev1, colored by sex
ggplot(cystic_fibrosis_data, aes(x = age, y = height, color = factor(sex))) +
  geom_point() +
  labs(title = "Relationship Between Age and Height, Differentiated by Gender",
       x = "Age in Years", y = "Height in Centimeters", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("green", "orange"))

ggplot(cystic_fibrosis_data, aes(x = age, y = weight, color = factor(sex))) +
  geom_point() +
  labs(title = "How Age Influences Weight Across Genders",
       x = "Age in Years", y = "Weight in Kilograms", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("darkred", "yellow"))

ggplot(cystic_fibrosis_data, aes(x = age, y = rv, color = factor(sex))) +
  geom_point() +
  labs(title = "Age vs Residual Volume by Gender",
       x = "Age (Years)", y = "RV (Residual Volume in mL)", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("purple", "lightblue"))

ggplot(cystic_fibrosis_data, aes(x = age, y = frc, color = factor(sex))) +
  geom_point() +
  labs(title = "Functional Residual Capacity Over Age, Segregated by Gender",
       x = "Age (Years)", y = "FRC (Functional Residual Capacity in mL)", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("cyan", "magenta"))

ggplot(cystic_fibrosis_data, aes(x = age, y = tlc, color = factor(sex))) +
  geom_point() +
  labs(title = "Total Lung Capacity vs Age, Analyzed by Gender",
       x = "Age in Years", y = "TLC (Total Lung Capacity in mL)", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("navy", "gold"))

ggplot(cystic_fibrosis_data, aes(x = age, y = pemax, color = factor(sex))) +
  geom_point() +
  labs(title = "Peak Expiratory Flow and Age Correlation by Gender",
       x = "Age (Years)", y = "PEMAX (Peak Expiratory Flow in L/min)", color = "Gender") +
  scale_color_manual(labels = c("Males", "Females"), values = c("black", "red"))


#b

library(ggplot2)

library(ggplot2)

# Height Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = height, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "Height Distribution by Sex", x = "Sex", y = "Height (cm)", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# Weight Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = weight, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "Weight Distribution by Sex", x = "Sex", y = "Weight (kg)", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# BMP Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = bmp, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "BMP Distribution by Sex", x = "Sex", y = "BMP", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# FEV1 Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = fev1, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "FEV1 Distribution by Sex", x = "Sex", y = "FEV1", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# RV Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = rv, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "RV Distribution by Sex", x = "Sex", y = "RV", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# FRC Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = frc, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "FRC Distribution by Sex", x = "Sex", y = "FRC", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# TLC Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = tlc, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "TLC Distribution by Sex", x = "Sex", y = "TLC", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# PEMAX Distribution by Sex
ggplot(cystic_fibrosis_data, aes(x = factor(sex), y = pemax, fill = factor(sex))) +
  geom_boxplot() +
  labs(title = "PEMAX Distribution by Sex", x = "Sex", y = "PEMAX", fill = "Sex") +
  scale_fill_manual(labels = c("Males", "Females"), values = c("purple", "yellow"))

# part 5

# Parameters
n <- 8   # Total number of patients (trials)
k <- 5   # Number of patients we're interested in recovering (successes)
p <- 0.87 # Probability of recovery (success)

# Calculating the probability
probability <- dbinom(k, n, p)

# Printing the result
probability


# part 6

# Parameters
lambda <- 6  # Average rate of emails per minute
k <- 8       # Number of emails we're interested in

# Calculating the probability
probability <- dpois(k, lambda)

# Printing the result
probability


# part 7

# Parameters
mean_sales <- 14600  # Mean daily sales in liters
sd_sales <- 2600     # Standard deviation of daily sales in liters
lower_bound <- 10000 # Lower bound of sales in liters we're interested in
upper_bound <- 20000 # Upper bound of sales in liters

# Calculate the probability of selling more than the lower bound
probability_more_than_lower_bound <- 1 - pnorm(lower_bound, mean_sales, sd_sales)

# Calculate the probability of selling less than or equal to the upper bound
probability_less_than_upper_bound <- pnorm(upper_bound, mean_sales, sd_sales)

# The probability of the sales being between the lower and upper bounds
probability_between_bounds <- probability_less_than_upper_bound - (1 - probability_more_than_lower_bound)

# Printing the result
probability_between_bounds

# check linda'swhatapp shared material.

# part 8
# A
#  temperature and sugar data
temperature <- c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
converted_sugar <- c(8.1, 7.8, 8.5, 9.8, 9.5, 8.9, 8.6, 10.2, 9.3, 9.2, 10.5)

# Perform linear regression analysis
regression_model <- lm(converted_sugar ~ temperature)

# Display the regression summary
summary(regression_model)

# Plotting the observed data
plot(temperature, converted_sugar, main = "Linear Regression Analysis",
     xlab = "Temperature", ylab = "Converted Sugar", pch = 19, col = "blue")

# Adding the regression line
abline(regression_model, col = "red", lwd = 2)

# Adding a legend
legend("topleft", legend=c("Observed data", "Fitted line"),
       col=c("blue", "red"), lwd=2, pch=19)


#part b

# Load the necessary libraries
library(ggplot2)

# Sample data vectors representing temperature and amount of converted sugar
temperature <- c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
converted_sugar <- c(8.1, 7.8, 8.5, 9.8, 9.5, 8.9, 8.6, 10.2, 9.3, 9.2, 10.5)

# Perform linear regression analysis
regression_model <- lm(converted_sugar ~ temperature)

# Plotting the observed data and the regression line
ggplot(data = data.frame(temperature, converted_sugar), aes(x = temperature, y = converted_sugar)) +
  geom_point() +
  geom_smooth(method = "lm", col = "red") +
  labs(title = "Linear Regression Analysis", x = "Temperature", y = "Converted Sugar")

# Linear interpolation to estimate the mean amount of sugar at temperature 1.75
interpolated_sugar <- approx(temperature, converted_sugar, xout = 1.75)$y

# Since we are only interested in the interpolated value, we do not need to take the mean of a single value.
# The interpolated value itself is our estimate.
mean_sugar <- mean(interpolated_sugar)

# Print the interpolated sugar amount at temperature 1.75
cat("The interpolated amount of converted sugar at temperature 1.75 is:", interpolated_sugar, "\n")

# Print the mean of the estimated sugar amount, which in this case is the same as the interpolated value
cat("The mean amount of converted sugar at temperature 1.75 is:", mean_sugar, "\n")



# part 9

# Sample data vectors (replace these with your actual data)
advert <- c(0, 10, 4, 5, 2, 7, 3, 6)
purchase <- c(4, 12, 5, 10, 1, 3, 4, 8)

# Calculating Pearson's correlation coefficient
correlation_coefficient <- cor(advert, purchase)

# Printing the coefficient
correlation_coefficient


#b
# Scatter plot to inspect the relationship
plot(advert, purchase, main = "Advertisements vs. Purchases", xlab = "Number of Advertisements", ylab = "Number of Purchases", pch = 19)

# part 10
# Example data frame for demonstration (assuming traffic speed data is available)
traffic_data <- data.frame(
  day_of_week = c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'),
  average_speed = c(70, 68, 69, 65, 60, 75, 78) # hypothetical average speeds
)

# Descriptive analysis: average speed by day of the week
summary(traffic_data$average_speed)

# Identify the day with the highest average speed
best_day_for_delivery <- traffic_data[which.max(traffic_data$average_speed), ]

# Printing the best day for delivery based on average speed
print(best_day_for_delivery)


