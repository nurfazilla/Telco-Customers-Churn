# -*- coding: utf-8 -*-
"""
EXPLORATORY DATA ANALYSIS
DATA: Telco Customer Churn
SOURCE; https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    
"""

# Linear algebra
import numpy as np

# Data processing
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statisical analysis
import scipy.stats as stats

# Data Exploratory
from exploration_utils import dataset_exploration

# Feature Scaling
from sklearn.preprocessing import StandardScaler

#*********************************************************************************************

df = pd.read_parquet("C:/Users/hp/Documents/Telco Customer Retention/telco_retention.parquet")

# Create a separate data for churned and non-churned customers
df_churn = df[df['Churn'] == 'Yes']
df_not_churn = df[df['Churn'] == 'No']
dataset_exploration(df_churn)
dataset_exploration(df_not_churn)



# Bivariate analysis - target and categorical independent vars
var_list = df[['gender', 'Partner', 'Dependents', 
               'PhoneService', 'MultipleLines', 'InternetService', 
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
               'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']]
for var in var_list:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=var, hue='Churn', palette='Paired')
    plt.title(f'Churn Count by {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.legend(title='Churned', loc='upper right', labels=['Not Churned', 'Churned'], bbox_to_anchor=(1.35, 1))
    plt.show()
    
variables = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for var in variables:
    crosstab = pd.crosstab(df['InternetService'], df[var])
    print(f"Crosstab of InternetService and {var}:\n", crosstab, "\n")


# Bivariate analysis - target and numerical independent vars
def plot_kde(variable):
    sns.kdeplot(df[variable], label='All Customers', fill=True)
    sns.kdeplot(df_churn[variable], label='Churned Customers', fill=True)
    sns.kdeplot(df_not_churn[variable], label='Non-Churned Customers', fill=True)
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.title(f'KDE of {variable} for All Customers, Churned Customers, and Non-Churned Customers')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
plot_kde('TotalCharges')
plot_kde('MonthlyCharges')
plot_kde('tenure')


# Bivariate analysis - categorical and Monthly Charges
var_list = df[['gender', 'Partner', 'Dependents', 
               'PhoneService', 'MultipleLines', 'InternetService', 
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
               'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']]

for var in var_list.columns:
    plt.figure(figsize=(10, 6))
    # Loop through the unique values of the categorical variable
    for value in df_not_churn[var].unique():
        subset = df_not_churn[df[var] == value]
        # Plotting the KDE
        sns.kdeplot(subset['MonthlyCharges'], label=value, shade=True)
        # Plotting the histogram
        plt.hist(subset['MonthlyCharges'], alpha=0.3, bins=30)

    plt.title(f'Monthly Charges Distribution by {var} for Non-Churned Customers')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Density')
    plt.legend(title=var)
    plt.show()

# Assuming 'Contract' and 'MonthlyCharges' are columns in your DataFrame df
contract_types = df['Contract'].unique()
# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, len(contract_types), figsize=(15, 6), sharey=True)
for i, contract in enumerate(contract_types):
    # Filter the DataFrame for each contract type
    subset = df_churn[df_churn['Contract'] == contract]
    # Plotting the histogram on each subplot
    axes[i].hist(subset['MonthlyCharges'], bins=30, alpha=0.7)
    axes[i].set_title(f'Contract: {contract}')
    axes[i].set_xlabel('Monthly Charges')
axes[0].set_ylabel('Frequency')
plt.suptitle('Monthly Charges Distribution by Contract Type for CHurned Customers')
plt.tight_layout()
plt.show()


# Correlation analysis
# One-hot encoding of categorical variables
df_encoded = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 
                                         'PhoneService', 'MultipleLines', 'InternetService', 
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                         'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'])
# Convert the 'Yes' and 'No' in the Churn column to 1 and 0 respectively
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Scaling for numerical variables
# Create a StandardScaler object
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])


# Calculate correlation with 'Churn' for each variable
correlation_with_churn = df_encoded.drop(['Churn', 'customerID'], axis=1).apply(lambda x: x.corr(df_encoded['Churn']))
# Sort values for better visualization
sorted_correlations = correlation_with_churn.sort_values(ascending=False)
# Plot
plt.figure(figsize=(15, 8))
bar_plot = sns.barplot(x=sorted_correlations.index, y=sorted_correlations.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.ylabel('Correlation with Churn')
plt.xlabel('Features')
plt.title('Correlation of Features with Churn')
# Adding data labels
for bar in bar_plot.patches:
    y_value = bar.get_height()
    x_value = bar.get_x() + bar.get_width() / 2
    label = round(y_value, 1)  # rounding to 2 decimal places
    bar_plot.text(x_value, y_value, label, ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()



# Multivariate analysis
# Analysis of tenure distribution for churned and non churned customers by gender
# For Male Customers
male_churn = df[(df['gender'] == 'Male') & (df['Churn'] == 'Yes')]['tenure']
male_no_churn = df[(df['gender'] == 'Male') & (df['Churn'] == 'No')]['tenure']
# For Female Customers
female_churn = df[(df['gender'] == 'Female') & (df['Churn'] == 'Yes')]['tenure']
female_no_churn = df[(df['gender'] == 'Female') & (df['Churn'] == 'No')]['tenure']

# Set up the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
# Plotting for Male Customers
sns.kdeplot(data=male_churn, ax=ax[0], label='Churned', shade=True)
sns.kdeplot(data=male_no_churn, ax=ax[0], label='Not Churned', shade=True)
ax[0].set_title('Distribution of Tenure for Male Customers')
ax[0].set_xlabel('Tenure')
ax[0].set_ylabel('Density')
ax[0].legend()
# Plotting for Female Customers
sns.kdeplot(data=female_churn, ax=ax[1], label='Churned', shade=True)
sns.kdeplot(data=female_no_churn, ax=ax[1], label='Not Churned', shade=True)
ax[1].set_title('Distribution of Tenure for Female Customers')
ax[1].set_xlabel('Tenure')
ax[1].set_ylabel('Density')
ax[1].legend()
# Display the plots
plt.tight_layout()
plt.show()


# Analysis of tenure distribution for churned and non churned customers by senior citizens
# For Senior Citizens
senior_churn = df[(df['SeniorCitizen'] == 1) & (df['Churn'] == 'Yes')]['tenure']
senior_no_churn = df[(df['SeniorCitizen'] == 1) & (df['Churn'] == 'No')]['tenure']
# For Non-Senior Citizens
non_senior_churn = df[(df['SeniorCitizen'] == 0) & (df['Churn'] == 'Yes')]['tenure']
non_senior_no_churn = df[(df['SeniorCitizen'] == 0) & (df['Churn'] == 'No')]['tenure']

# Set up the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
# Plotting for Male Customers
sns.kdeplot(data=senior_churn, ax=ax[0], label='Churned', shade=True)
sns.kdeplot(data=senior_no_churn, ax=ax[0], label='Not Churned', shade=True)
ax[0].set_title('Distribution of Tenure for Senior Citizen Customers')
ax[0].set_xlabel('Tenure')
ax[0].set_ylabel('Density')
ax[0].legend()
# Plotting for Female Customers
sns.kdeplot(data=non_senior_churn, ax=ax[1], label='Churned', shade=True)
sns.kdeplot(data=non_senior_no_churn, ax=ax[1], label='Not Churned', shade=True)
ax[1].set_title('Distribution of Tenure for Non-Senior Customers')
ax[1].set_xlabel('Tenure')
ax[1].set_ylabel('Density')
ax[1].legend()
# Display the plots
plt.tight_layout()
plt.show()



#Other analysis - Continuation ratio
# Count the number of customers for each tenure
total_customers = df['tenure'].value_counts().sort_index()
# Count the number of churned customers for each tenure
churned_customers = df[df['Churn'] == 'Yes']['tenure'].value_counts().sort_index()
# Calculate the continuation ratio
continuation_ratio = (total_customers - churned_customers) / total_customers
# Visualize the continuation ratios
plt.figure(figsize=(15, 7))
continuation_ratio.plot()
plt.title('Continuation Ratios by Tenure Length')
plt.xlabel('Tenure (months)')
plt.ylabel('Continuation Ratio')
plt.grid(True)
plt.show()


# Save file to parquet
df.to_parquet('telco_retentionv1.parquet')

