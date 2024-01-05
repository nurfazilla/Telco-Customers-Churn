# -*- coding: utf-8 -*-
"""
PRELIMNARY OBSERVATIONS
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

# Data Exploratory
from exploration_utils import dataset_exploration


#*********************************************************************************************


df = pd.read_csv("C:/Users/hp/Documents/Telco Customer Retention/WA_Fn-UseC_-Telco-Customer-Churn.csv")
dataset_exploration(df)

# Make a copy
df = df.copy()  

# Change correct data types
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values (tenure =0 and totalcharges=0)
df1 = df[df['tenure'] != 0]

# To replace 'No internet service' and 'No phone service' with 'No'
columns_to_update = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']

# Replace 'No internet service' and 'No phone service' with 'No' in the specified columns
for col in columns_to_update:
    df1[col] = df1[col].replace({'No internet service': 'No', 'No phone service': 'No'}) 

# Save file to parquet
df1.to_parquet('telco_retention.parquet')



