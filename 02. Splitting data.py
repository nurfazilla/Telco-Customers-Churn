# -*- coding: utf-8 -*-
"""
SPLITTING DATA TO TRAIN AND TEST TEST
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

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#*********************************************************************************************
# SPLIT DATA
df = pd.read_parquet("C:/Users/hp/Documents/Telco Customer Retention/telco_retentionv1.parquet")

# Split the test data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Combine the X and Y for training and testing data
df_training = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

df_training = df_training.drop(['gender','PhoneService','MultipleLines','StreamingTV','StreamingMovies'], axis=1)
df_test = df_test.drop(['gender','PhoneService','MultipleLines','StreamingTV','StreamingMovies'], axis=1)

# Save file to parquet
df_training.to_parquet('telco_retention_training.parquet')
df_test.to_parquet('telco_retention_test.parquet')
