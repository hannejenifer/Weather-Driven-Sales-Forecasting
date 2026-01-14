pip install pandas scikit-learn matplotlib seaborn

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from google.colab import files
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Upload the datasets
uploaded = files.upload()
uploaded = files.upload()

# Load the datasets
weather_df = pd.read_csv('Chennai_1990_2022_Madras.csv')
price_df = pd.read_csv('FAOSTAT_data_en_10-5-2024.csv')

# Step 2: Check the columns, print in a readable format, and find common columns
def print_columns(df, df_name):
    print(f"\n{df_name} Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

# Function to find and print common columns
def print_common_columns(df1, df2, df1_name, df2_name):
    common_cols = set(df1.columns).intersection(set(df2.columns))
    print(f"\nCommon Columns between {df1_name} and {df2_name}:")
    if common_cols:
        for i, col in enumerate(common_cols, 1):
            print(f"{i}. {col}")
    else:
        print("No common columns found.")

# Print the columns of both datasets
print_columns(weather_df, "Weather Data")
print_columns(price_df, "FAOSTAT Price Data")

# Print common columns
print_common_columns(weather_df, price_df, "Weather Data", "FAOSTAT Price Data")

# Load the datasets
weather_df = pd.read_csv('Chennai_1990_2022_Madras.csv')
price_df = pd.read_csv('FAOSTAT_data_en_10-5-2024.csv')

# Extract year from the existing 'time' column
weather_df['Year'] = pd.to_datetime(weather_df['time'], format='%d-%m-%Y').dt.year

# Merge the datasets on the 'Year' column
merged_df = pd.merge(price_df, weather_df, on='Year')

# Display the first few rows of the merged dataset in a tabular format
print("Merged Dataset:")
print(tabulate(merged_df.head(), headers='keys', tablefmt='pretty'))

# Load the datasets
weather_df = pd.read_csv('Chennai_1990_2022_Madras.csv')
price_df = pd.read_csv('FAOSTAT_data_en_10-5-2024.csv')

# Extract year from the existing 'time' column
weather_df['Year'] = pd.to_datetime(weather_df['time'], format='%d-%m-%Y').dt.year

# Merge the datasets on the 'Year' column
merged_df = pd.merge(price_df, weather_df, on='Year')

# Step 5: Filter the dataset for years 2013 to 2022
train_data = merged_df[(merged_df['Year'] >= 2013) & (merged_df['Year'] <= 2022)]

# Display the filtered dataset in a tabular format
print("Filtered Training Data (2013-2022):")
print(tabulate(train_data.head(), headers='keys', tablefmt='pretty'))

# Optionally, save the merged dataset to a CSV file
merged_df.to_csv('merged_dataset.csv', index=False)

# Filter the dataset for years 2013 to 2022
train_data = merged_df[(merged_df['Year'] >= 2013) & (merged_df['Year'] <= 2022)]

# Define the features (weather data) and target variable (price index)
X_train = train_data[['tavg', 'tmin', 'tmax', 'prcp']]
y_train = train_data['Value']

# Split the data into training and validation sets (80% train, 20% validation)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Validate the model
y_val_pred = model.predict(X_val_split)
mse = mean_squared_error(y_val_split, y_val_pred)
print(f'Mean Squared Error (Validation): {mse:.2f}')

# Prepare to predict prices for 2024 based on expected weather conditions
weather_2024_avg = {
    'tavg': [29.06],
    'tmin': [26.63],
    'tmax': [31.48],
    'prcp': [116.42]
}
weather_2024_df = pd.DataFrame(weather_2024_avg)

# Predict the price for 2024
predicted_price_2024 = model.predict(weather_2024_df)
print(f'Predicted Price for 2024: {predicted_price_2024[0]:.2f}')

# Load your merged dataset
merged_df = pd.read_csv('merged_dataset.csv')

# List of unique ingredients
ingredients = merged_df['Item'].unique()

# Create a DataFrame to store the predictions
ingredient_predictions = []

# Loop through each ingredient to train individual models
for ingredient in ingredients:
    # Filter data for the current ingredient
    ingredient_data = merged_df[merged_df['Item'] == ingredient]

    # Select weather features and price
    X = ingredient_data[['tavg', 'tmin', 'tmax', 'prcp']]
    y = ingredient_data['Value']

    # Drop rows with missing values
    X = X[y.notna()]
    y = y.dropna()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict 2024 prices using average 2024 weather data
    weather_2024_avg = {'tavg': [29.06], 'tmin': [26.63], 'tmax': [31.48], 'prcp': [116.42]}  # Based on your input
    weather_2024_df = pd.DataFrame(weather_2024_avg)
    predicted_price = model.predict(weather_2024_df)

    # Store the ingredient and predicted price
    ingredient_predictions.append({
        'Ingredient': ingredient,
        'Predicted Price 2024': predicted_price[0]
    })

# Convert predictions into a DataFrame
predictions_df = pd.DataFrame(ingredient_predictions)

# Sort to find highest and lowest prices
sorted_predictions = predictions_df.sort_values(by='Predicted Price 2024')
print(sorted_predictions)

# Assuming importances and X are defined
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by Importance
sorted_feature_importance = feature_importance_df.sort_values(by='Importance', ascending=False)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=sorted_feature_importance, palette='viridis')

# Add title and labels
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Show the plot
plt.show()

# Assuming you have already trained your model and have X_test and y_test
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Prepare data for plotting
metrics = ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R² Score']
values = [mae, mse, r2]

# Create a bar plot
plt.figure(figsize=(8, 5))
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.bar(metrics, values, color=bar_colors)

# Add title and labels
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
plt.ylim(0, max(values) * 1.2)  # Set y-axis limit to be a little higher than the max value

# Display the value on top of the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

# Show the plot
plt.show()

# Create a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, color='blue', label='Actual Prices', fill=True, alpha=0.5)
sns.kdeplot(y_pred, color='orange', label='Predicted Prices', fill=True, alpha=0.5)

# Set axis labels and title
plt.xlabel('Prices (INR)', fontsize=14)
plt.title('Density Plot of Actual vs Predicted Prices', fontsize=16)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Calculate errors
errors = y_test - y_pred

# Create an error distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='purple', alpha=0.7)

# Set axis labels and title
plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Prediction Errors', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Create the plot
plt.figure(figsize=(10, 6))

# Create a histogram with a KDE overlay using light green
sns.histplot(residuals, bins=30, kde=True, color='darkgreen', edgecolor='black', alpha=0.6)

# Calculate and display mean and standard deviation
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

# Draw lines for mean and zero residual
plt.axvline(x=mean_residual, color='orange', linestyle='--', label='Mean = {:.2f}'.format(mean_residual))
plt.axvline(x=0, color='red', linestyle='--', label='Zero Residual')

# Set axis labels and title
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Load the dataset
data = pd.read_csv('/content/updated_foods_with_imported_and_sensitivity.csv')

# Create a new DataFrame for heatmap purposes
# Assign numerical values for each price sensitivity level
sensitivity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
data['Sensitivity Value'] = data['Price Sensitivity'].map(sensitivity_mapping)

# Create a pivot table for the heatmap
heatmap_data = data.pivot_table(index='Item', values='Sensitivity Value', aggfunc='mean')

# Sort the data to ensure high sensitivity values are at the top
heatmap_data = heatmap_data.sort_values(by='Sensitivity Value', ascending=False)

# Define a color palette that gradually transitions from sky blue to yellow to red
cmap = sns.color_palette("RdYlBu_r", as_cmap=True)  # Reversed Red-Yellow-Blue palette for smoother transition

# Increase figure size to make the plot less congested
plt.figure(figsize=(12, 15))
sns.heatmap(
    heatmap_data,
    annot=False,  # Disable annotations
    cmap=cmap,  # Use the defined color palette
    linewidths=0.5,
    cbar_kws={'label': 'Price Sensitivity', 'shrink': 0.5}  # Add color bar label)
plt.title('Heatmap of Price Sensitivity for Items', fontsize=18, pad=20)
plt.ylabel('Item', fontsize=14, labelpad=10)
plt.xlabel('Price Sensitivity', fontsize=14)  # Single x-axis label
plt.xticks(ticks=[0], labels=['Price Sensitivity'], rotation=0)
plt.tight_layout()
plt.show()

# Count the occurrences of imported vs non-imported items
import_counts = data['Imported'].value_counts()

# Create the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=import_counts.index, y=import_counts.values, palette='viridis')

# Customize the appearance
plt.title('Count of Imported vs Non-Imported Items')
plt.xlabel('Imported Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')  # Add grid lines for better readability
plt.tight_layout()

# Display the plot
plt.show()

warnings.filterwarnings("ignore")
data = pd.read_csv('/content/updated_foods_with_imported_and_sensitivity.csv')

# Filter the dataset to include only imported items
imported_data = data[data['Imported'] == 'Yes']  # Adjust if your 'Imported' column has different values

# Calculate the average values for each price sensitivity category
average_values = imported_data.groupby('Price Sensitivity')['Value'].mean().reset_index()

# Create the bar chart with specified colors for each category
plt.figure(figsize=(10, 6))
sns.barplot(x='Price Sensitivity', y='Value', data=average_values, palette=['blue', 'green', 'red'])

# Customize the appearance
plt.title('Imported Items by Price Sensitivity')
plt.xlabel('Price Sensitivity')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.grid(axis='y')  # Add grid lines for better readability
plt.tight_layout()

# Display the plot
plt.show()

# Filter the dataset to include only non-imported items
non_imported_data = data[data['Imported'] == 'No']

# Calculate the average value for non-imported items grouped by price sensitivity
average_values = non_imported_data.groupby('Price Sensitivity')['Value'].mean().reindex(['Low', 'Medium', 'High']).reset_index()

# Create the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Price Sensitivity', y='Value', data=average_values, palette=['purple', 'orange', 'red'])

# Customize the appearance
plt.title('Non-Imported Items by Price Sensitivity')
plt.xlabel('Price Sensitivity')
plt.ylabel('Average Value')
plt.grid(axis='y')  # Add grid lines for better readability
plt.tight_layout()

# Display the plot
plt.show()

# Group by year and calculate the total imported values
year_imported_sum = data.groupby(['Year', 'Imported'])['Value'].sum().unstack()

# Create the line plot
plt.figure(figsize=(12, 6))

# Plot each line separately to customize colors
year_imported_sum['Yes'].plot(kind='line', marker='o', color='red', linewidth=2, label='Imported (Yes)')
year_imported_sum['No'].plot(kind='line', marker='o', color='orange', linewidth=2, label='Imported (No)')

# Customize the appearance
plt.title('Trend of Total Imported Value Over Years')
plt.xlabel('Year')
plt.ylabel('Total Imported Value')
plt.xticks(rotation=45)
plt.grid()
plt.legend()  # Show the legend
plt.tight_layout()

# Display the plot
plt.show()

year_value_data = data.groupby('Year').agg({'Value': 'sum'}).reset_index()

plt.figure(figsize=(15, 8))
sns.barplot(data=year_value_data, x='Year', y='Value', palette='viridis')

plt.title('Total Value of Imported Items Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Value')
plt.xticks(rotation=45)
plt.show()

# Load the dataset from the specified path
data = pd.read_csv('/content/highly_perishable_foods.csv')

# Group by Year and Item, summing the Value
year_item_value = data.groupby(['Year', 'Item'])['Value'].sum().unstack()

# Create the line plot
plt.figure(figsize=(12, 6))
year_item_value.plot(kind='line', marker='o', figsize=(14, 8))

# Customize the appearance
plt.title('Trend of Quantity of Import of Items Over Years - Highly Perishable Goods')
plt.xlabel('Year')
plt.ylabel('Quantity of Goods')
plt.xticks(rotation=45)
plt.grid()
plt.legend(title='Item')
plt.tight_layout()

# Display the plot
plt.show()

# Create a FacetGrid for line plots
g = sns.FacetGrid(data, col='Item', col_wrap=3, height=4, sharey=False)
g.map(sns.lineplot, 'Year', 'Value', marker='o')

# Customize the appearance
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Year", "Quantity")
g.add_legend()

# Adjust the layout
plt.tight_layout()
plt.show()
https://colab.research.google.com/drive/1Fo_1FfEkGG7vIJRPOYWITyCyRWerS9lJ#scrollTo=eQ-6XSl-legR

# Group by Year and Item, summing the Value
year_item_value = data.groupby(['Year', 'Item'])['Value'].sum().unstack()

# Create the line plot
plt.figure(figsize=(12, 6))
year_item_value.plot(kind='line', marker='o', figsize=(14, 8))

# Customize the appearance
plt.title('Trend of Quantity of Import of Items Over Years - Perishable Goods')
plt.xlabel('Year')
plt.ylabel('Quantity of goods')
plt.xticks(rotation=45)
plt.grid()
plt.legend(title='Item')
plt.tight_layout()

# Display the plot
plt.show()

# Create a FacetGrid for line plots
g = sns.FacetGrid(data, col='Item', col_wrap=3, height=4, sharey=False)
g.map(sns.lineplot, 'Year', 'Value', marker='o')

# Customize the appearance
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Year", "Quantity")
g.add_legend()

# Adjust the layout
plt.tight_layout()
plt.show()

# Load the dataset from the specified path
data = pd.read_csv('/content/non_perishable_foods.csv')

# Group by Year and Item, summing the Value
year_item_value = data.groupby(['Year', 'Item'])['Value'].sum().unstack()

# Create the line plot
plt.figure(figsize=(12, 6))
year_item_value.plot(kind='line', marker='o', figsize=(14, 8))

# Customize the appearance
plt.title('Trend of Quantity of Import of Items Over Years - Non Perishable Goods')
plt.xlabel('Year')
plt.ylabel('Quantity of Goods')
plt.xticks(rotation=45)
plt.grid()
plt.legend(title='Item')
plt.tight_layout()

# Display the plot
plt.show()

# Create a FacetGrid for line plots
g = sns.FacetGrid(data, col='Item', col_wrap=3, height=4, sharey=False)
g.map(sns.lineplot, 'Year', 'Value', marker='o')

# Customize the appearance
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Year", "Quantity")
g.add_legend()

# Adjust the layout
plt.tight_layout()
plt.show()
