#%% md
# 1. Importing Libraries

**Explanation**:
- **pandas** and **numpy**: For data manipulation and numerical operations.
- **matplotlib** and **seaborn**: For data visualization.
- **sklearn.preprocessing.StandardScaler**: To scale features before feeding them into the model.
- **sklearn.model_selection.train_test_split**: To split the dataset into training and testing subsets.
- **sklearn.ensemble.RandomForestRegressor**: For training a Random Forest model.
- **sklearn.linear_model.LinearRegression**: For training a Linear Regression model.
- **sklearn.metrics.mean_squared_error, r2_score**: To evaluate the models using metrics.
- **shap** and **lime.lime_tabular**: For model interpretability.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap
import lime.lime_tabular

#%% md
# 2. Loading and Preparing the Crime Data

**Explanation**:
- **crime_df**: Load the CSV file containing crime data.
- **'Month' to datetime**: Convert the 'Month' column to a datetime object for easier filtering.
    - **Filter the DataFrame**: Keep only the records from 2021 related to "Violence and sexual offences".
- **Modify 'LSOA name'**: Truncate the last 5 characters from 'LSOA name' to generalize it.
- **Rename column**: Change the column name to 'lower_tier_local_authorities'.
- **Aggregate Data**: Group by 'Month' and 'lower_tier_local_authorities', counting the number of crimes.

#%%
# Load the dataset containing crime data
crime_data_path = "/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/UK_Police_Street_Crime_2018-10-01_to_2021_09_31.csv"
crime_df = pd.read_csv(crime_data_path)

# Convert 'Month' to datetime and filter for 2021 and specific crime type
crime_df['Month'] = pd.to_datetime(crime_df['Month'])
filtered_crime_df = crime_df[(crime_df['Month'].dt.year == 2021) &
                             (crime_df['Crime type'] == "Violence and sexual offences")]

# Modify 'LSOA name' to remove the last 5 characters
filtered_crime_df['LSOA name'] = filtered_crime_df['LSOA name'].str[:-5]

# Rename 'LSOA name' to 'lower_tier_local_authorities'
filtered_crime_df.rename(columns={'LSOA name': 'lower_tier_local_authorities'}, inplace=True)

# Aggregate the data by 'Month' and 'lower_tier_local_authorities' and count occurrences
aggregated_crime_data = filtered_crime_df.groupby(['Month', 'lower_tier_local_authorities']).size().reset_index(name='crime_count')

#%% md
# 3. Loading and Cleaning the Additional Dataset

**Explanation**:
- **file_path**: Load another dataset containing demographic and socio-economic data.
- **columns_to_drop**: Identify columns containing redundant 'lower_tier_local_authorities' labels.
- **Drop columns**: Remove all but one 'lower_tier_local_authorities' column to avoid duplication.
- **Rename column**: Standardize the column name for consistency.

#%%
# Load datasets
ethnic_data = pd.read_csv("/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Ethnic_Group_Data_Transformed.csv")
economy_status_data = pd.read_csv("/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Economy_Status_Data_Transformed.csv")
employment_history_data = pd.read_csv("/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Employment_History_Data_Transformed.csv")
education_level_data = pd.read_csv("/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Education_Level_Data_Transformed.csv")
migrant_data = pd.read_csv("/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Migrant_Data_Transformed.csv")

# Rename the locality code columns to 'area_code'
ethnic_data.rename(columns={'ethnic_lower_tier_local_authorities_code': 'area_code'}, inplace=True)
economy_status_data.rename(columns={'economy_lower_tier_local_authorities_code': 'area_code'}, inplace=True)
employment_history_data.rename(columns={'employment_lower_tier_local_authorities_code': 'area_code'}, inplace=True)
education_level_data.rename(columns={'education_lower_tier_local_authorities_code': 'area_code'}, inplace=True)
migrant_data.rename(columns={'migrant_lower_tier_local_authorities_code': 'area_code'}, inplace=True)

# Merge all datasets on the 'area_code' column
merged_data = ethnic_data.merge(economy_status_data, on='area_code', how='outer')
merged_data = merged_data.merge(employment_history_data, on='area_code', how='outer')
merged_data = merged_data.merge(education_level_data, on='area_code', how='outer')
merged_data = merged_data.merge(migrant_data, on='area_code', how='outer')

# Save the merged dataset to a CSV file (optional)
# merged_data.to_csv('path_to/merged_dataset.csv', index=False)
merged_data
#%%
# Load the additional dataset
# file_path = '/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/Merged_Dataset.csv'
# additional_df = pd.read_csv(file_path)
additional_df = merged_data.copy()
# Identify and drop redundant 'lower_tier_local_authorities' columns
columns_to_drop = [col for col in additional_df.columns if 'lower_tier_local_authorities' in col]
columns_to_drop.remove('ethnic_lower_tier_local_authorities')  # Keep one relevant column

# Drop the identified columns
df_cleaned = additional_df.drop(columns=columns_to_drop)

# Rename the relevant column to 'lower_tier_local_authorities'
df_cleaned.rename(columns={'ethnic_lower_tier_local_authorities': 'lower_tier_local_authorities'}, inplace=True)

#%% md
# 4. Merging the Datasets

**Explanation**:
- **Merge DataFrames**: Combine the crime data and the cleaned demographic data based on 'lower_tier_local_authorities'.
- **Display Info**: Print the first few rows and the structure of the merged DataFrame to verify the merge.

#%%
# Merge the crime data with the cleaned dataset on 'lower_tier_local_authorities'
merged_data = aggregated_crime_data.merge(df_cleaned, on='lower_tier_local_authorities', how='inner')

# Display basic information and initial rows of the merged dataset
print(merged_data.head())
print(merged_data.info())

#%% md
# 5. Preprocessing and Feature Engineering

**Explanation**:
- **Drop columns**: Remove irrelevant columns like 'lower_tier_local_authorities' and 'area_code'.
- **Column mapping**: Simplify column names for easier reference.
    - **Select features**: Choose the most correlated features within each category (e.g., Migrant, Education) with the target variable, `crime_count`.
- **Scale Features**: Standardize the selected features for modeling.
    - **Convert 'Month'**: Transform the 'Month' column into a numeric format for use in models.

#%%
# Create a copy for processing
data = merged_data.copy()

# Drop columns not needed for analysis
data.drop(columns=['lower_tier_local_authorities', 'area_code'], inplace=True)

# Mapping new column names for easier reference
new_column_names = {
    'ethnic_asian_asian_british_or_asian_welsh_bangladeshi': 'Ethnic_Asian_Bangladeshi',
    'ethnic_asian_asian_british_or_asian_welsh_chinese': 'Ethnic_Asian_Chinese',
    'ethnic_asian_asian_british_or_asian_welsh_indian': 'Ethnic_Asian_Indian',
    'ethnic_asian_asian_british_or_asian_welsh_other_asian': 'Ethnic_Asian_Other',
    'ethnic_asian_asian_british_or_asian_welsh_pakistani': 'Ethnic_Asian_Pakistani',
    'ethnic_black_black_british_black_welsh_caribbean_or_african_african': 'Ethnic_Black_African',
    'ethnic_black_black_british_black_welsh_caribbean_or_african_caribbean': 'Ethnic_Black_Caribbean',
    'ethnic_black_black_british_black_welsh_caribbean_or_african_other_black': 'Ethnic_Black_Other',
    'ethnic_mixed_or_multiple_ethnic_groups_other_mixed_or_multiple_ethnic_groups': 'Ethnic_Mixed_Other',
    'ethnic_mixed_or_multiple_ethnic_groups_white_and_asian': 'Ethnic_Mixed_White_Asian',
    'ethnic_mixed_or_multiple_ethnic_groups_white_and_black_african': 'Ethnic_Mixed_White_Black_African',
    'ethnic_mixed_or_multiple_ethnic_groups_white_and_black_caribbean': 'Ethnic_Mixed_White_Black_Caribbean',
    'ethnic_other_ethnic_group_any_other_ethnic_group': 'Ethnic_Other',
    'ethnic_other_ethnic_group_arab': 'Ethnic_Arab',
    'ethnic_white_english_welsh_scottish_northern_irish_or_british': 'Ethnic_White_British',
    'ethnic_white_gypsy_or_irish_traveller': 'Ethnic_White_Gypsy_Traveller',
    'ethnic_white_irish': 'Ethnic_White_Irish',
    'ethnic_white_other_white': 'Ethnic_White_Other',
    'ethnic_white_roma': 'Ethnic_White_Roma',
    'economy_employed': 'Economy_Employed',
    'economy_not_employed': 'Economy_Not_Employed',
    'employment_not_in_employment_never_worked': 'Employment_Never_Worked',
    'employment_not_in_employment_not_worked_in_the_last_12_months': 'Employment_Not_Worked_12M',
    'employment_not_in_employment_worked_in_the_last_12_months': 'Employment_Worked_12M',
    'education_level_1_and_entry_level_qualifications_1_to_4_gcses_grade_a*_to_c_any_gcses_at_other_grades_o_levels_or_cses_any_grades_1_as_level_nvq_level_1_foundation_gnvq_basic_or_essential_skills': 'Education_Level_1',
    'education_level_2_qualifications_5_or_more_gcses_a*_to_c_or_9_to_4_o_levels_passes_cses_grade_1_school_certification_1_a_level_2_to_3_as_levels_vces_intermediate_or_higher_diploma_welsh_baccalaureate_intermediate_diploma_nvq_level_2_intermediate_gnvq_city_and_guilds_craft_btec_first_or_general_diploma_rsa_diploma': 'Education_Level_2',
    'education_level_3_qualifications_2_or_more_a_levels_or_vces_4_or_more_as_levels_higher_school_certificate_progression_or_advanced_diploma_welsh_baccalaureate_advance_diploma_nvq_level_3;_advanced_gnvq_city_and_guilds_advanced_craft_onc_ond_btec_national_rsa_advanced_diploma': 'Education_Level_3',
    'education_level_4_qualifications_or_above_degree_ba_bsc_higher_degree_ma_phd_pgce_nvq_level_4_to_5_hnc_hnd_rsa_higher_diploma_btec_higher_level_professional_qualifications_for_example_teaching_nursing_accountancy': 'Education_Level_4+',
    'education_no_qualifications': 'Education_No_Qualifications',
    'education_other_apprenticeships_vocational_or_work-related_qualifications_other_qualifications_achieved_in_england_or_wales_qualifications_achieved_outside_england_or_wales_equivalent_not_stated_or_unknown': 'Education_Other_Qualifications',
    'migrant_address_one_year_ago_is_student_term-time_or_boarding_school_address_in_the_uk': 'Migrant_Student_UK',
    'migrant_address_one_year_ago_is_the_same_as_the_address_of_enumeration': 'Migrant_Same_Address',
    'migrant_does_not_apply': 'Migrant_NA',
    'migrant_migrant_from_outside_the_uk_address_one_year_ago_was_outside_the_uk': 'Migrant_Outside_UK',
    'migrant_migrant_from_within_the_uk_address_one_year_ago_was_in_the_uk': 'Migrant_Within_UK'
}

# Rename columns according to the mapping
data.rename(columns=new_column_names, inplace=True)

# Select columns for further analysis
prefixes = ['Migrant', 'Education', 'Economy', 'Ethnic', 'Employment']
selected_columns = []

for prefix in prefixes:
    prefix_columns = [col for col in data.columns if col.startswith(prefix)]
    corr_with_target = data[prefix_columns].corrwith(data['crime_count']).abs()
    best_column = corr_with_target.idxmax()
    selected_columns.append(best_column)

# No need to reload the data, proceed with the selected columns
selected_columns_with_month = ['Month'] + selected_columns + ['crime_count']
final_data = data[selected_columns_with_month]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(final_data.drop(columns=['Month', 'crime_count']))
scaled_features_df = pd.DataFrame(scaled_features, columns=selected_columns)

# # Combine scaled features with 'Month' and 'crime_count'
# final_data = pd.concat([final_data[['Month']], scaled_features_df, final_data['crime_count']], axis=1)
#
# # Convert 'Month' to a numerical format (Unix timestamp)
# final_data['Month'] = pd.to_datetime(final_data['Month']).astype(int) // 10**9

# Extract year and month from the 'Month' column and treat them as categorical
# Extract month from the 'Month' column and treat it as categorical
final_data['Month'] = pd.to_datetime(final_data['Month']).dt.month.astype('category')

# Combine scaled features with 'Month' and 'crime_count'
final_data = pd.concat([final_data[['Month']], scaled_features_df, final_data['crime_count']], axis=1)

#%%
final_data
#%% md
# 6. Correlation Matrix Visualization

**Explanation**:
- **Correlation Matrix**: Display the correlation matrix to understand the relationships between features and the target variable.
- **Heatmap**: Visual representation using Seaborn's heatmap for easier interpretation.

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
corr_matrix = final_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap with annotations and a stylish color map
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'aspect': 20, 'pad': 0.02})

# Customize the plot with a title, better layout, and larger font sizes
plt.title('Correlation Matrix of Features and Crime Count', fontsize=18, pad=20)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)

# Remove the top and right spines for a cleaner look
sns.despine()

# Adjust layout for better spacing
plt.tight_layout()

# Display the heatmap
plt.show()

#%% md
# 7. Train-Test Split and Model Training

**Explanation**:
- **Train-test split**: Split the data into 70% training and 30% testing subsets.
- **Model Training**: Train both a Random Forest model and a Linear Regression model on the training data.

#%%
# Split the data into training and testing sets
X = final_data.drop(columns=['crime_count'])
y = final_data['crime_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#%% md
# 8. Model Evaluation

**Explanation**:
- **Predict**: Generate predictions for the test set using both models.
- **Evaluate**: Calculate and print the Mean Squared Error (MSE) and R-squared (R²) values for both models to assess performance.

#%%
# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Predict and evaluate Linear Regression
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Print evaluation metrics
print("Random Forest Evaluation:")
print(f"R-squared (R²): {r2_rf:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}\n")

print("Linear Regression Evaluation:")
print(f"R-squared (R²): {r2_linear:.4f}")
print(f"Mean Squared Error (MSE): {mse_linear:.4f}\n")

#%% md
# 9. Visualization of Predictions vs Actual Values

**Explanation**:
- **Plot**: Create scatter plots comparing actual vs predicted values for both models to visually assess their performance.

#%%
# Visualization: Predicted vs Actual for Random Forest and Linear Regression
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual Crime Count')
plt.ylabel('Predicted Crime Count')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_linear, alpha=0.6, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Crime Count')
plt.ylabel('Predicted Crime Count')

plt.tight_layout()
plt.show()

#%% md
# 10. Feature Importance Visualization

**Explanation**:
- **Feature Importance**: Calculate and visualize feature importance from the Random Forest model and Linear Regression model coefficients.

#%%
# Random Forest Feature Importance
importances_rf = rf_model.feature_importances_
importance_df_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances_rf}).sort_values(by='Importance', ascending=False)

# Linear Regression Coefficients
coefficients_lr = linear_model.coef_
importance_df_lr = pd.DataFrame({'Feature': X_train.columns, 'Importance': coefficients_lr}).sort_values(by='Importance', ascending=False)

# Visualization of Feature Importances
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(importance_df_rf['Feature'], importance_df_rf['Importance'], color='skyblue')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')

plt.subplot(1, 2, 2)
plt.barh(importance_df_lr['Feature'], abs(importance_df_lr['Importance']), color='lightcoral')
plt.title('Linear Regression Feature Importance')
plt.xlabel('Absolute Coefficient Value')

plt.tight_layout()
plt.show()

#%% md
# 11. SHAP Values for Interpretation

**Explanation**:
- **SHAP Values**: Use SHAP to interpret feature importance and impact on predictions for both the Random Forest and Linear Regression models.

#%%
# SHAP interpretation for Random Forest
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)
shap.summary_plot(shap_values_rf, X_test, plot_type="bar")

# SHAP interpretation for Linear Regression
explainer_lr = shap.LinearExplainer(linear_model, X_train)
shap_values_lr = explainer_lr.shap_values(X_test)
shap.summary_plot(shap_values_lr, X_test, plot_type="bar")

#%%

#%% md
# 12. LIME for Interpretation

**Explanation**:
- **LIME**: Generate local interpretable model-agnostic explanations using LIME for both models on a specific instance from the test set.

#%%
# LIME interpretation for Random Forest
explainer_rf_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    mode='regression'
)
instance_to_explain = X_test.iloc[0]
explanation_rf = explainer_rf_lime.explain_instance(
    data_row=instance_to_explain.values,
    predict_fn=rf_model.predict
)
explanation_rf.show_in_notebook(show_table=True)

# LIME interpretation for Linear Regression
explainer_lr_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    mode='regression'
)
explanation_lr = explainer_lr_lime.explain_instance(
    data_row=instance_to_explain.values,
    predict_fn=linear_model.predict
)
explanation_lr.show_in_notebook(show_table=True)

# Save LIME interpretation to HTML
rf_html_file = 'lime_rf_explanation.html'
explanation_rf.save_to_file(rf_html_file)
# Save LIME interpretation to HTML
lr_html_file = 'lime_lr_explanation.html'
explanation_lr.save_to_file(lr_html_file)
