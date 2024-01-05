# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:04:37 2024

@author: hp
"""

# Linear algebra
import numpy as np
# Data processing
import pandas as pd
# Data visualization
import matplotlib.pyplot as plt


# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
    OrdinalEncoder,
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.compose import make_column_selector, ColumnTransformer
from imblearn.over_sampling import (
    KMeansSMOTE,
    ADASYN,
    RandomOverSampler,
    SMOTE,
    SVMSMOTE,
)

# Data Exploratory
from exploration_utils import dataset_exploration

#*********************************************************************************************

df_train = pd.read_parquet("C:/Users/hp/Documents/Telco Customer Retention/telco_retention_training.parquet")
df_test = pd.read_parquet("C:/Users/hp/Documents/Telco Customer Retention/telco_retention_test.parquet")

####### 1. Splitting between train and validation

X = df_train.drop(['Churn','customerID'], axis=1)
y = df_train['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


####### 2. Data preperation for machine learning

# Define the columns to be scaled and encoded
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.drop('SeniorCitizen')
categorical_cols = X_train.select_dtypes(include=['object']).columns
# Create a ColumnTransformer object
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)
# Fit and transform the training DataFrame
X_train_encoded = preprocessor.fit_transform(X_train)
# Transform the test DataFrame (do not fit)
X_test_encoded = preprocessor.transform(X_test)

# Convert the encoded training data back to a DataFrame
encoded_columns = preprocessor.get_feature_names_out()
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns)
# Reset the index if necessary
X_train_encoded_df.reset_index(drop=True, inplace=True)
# Remove 'num__' and 'cat__' prefixes from the column names for the training set
new_column_names_train = [col.split('__')[-1] for col in encoded_columns]
X_train_encoded_df.columns = new_column_names_train

# Convert the encoded test data back to a DataFrame
# Note: Use the same encoded_columns as from the training data for consistency
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns)
# Reset the index if necessary
X_test_encoded_df.reset_index(drop=True, inplace=True)
# Remove 'num__' and 'cat__' prefixes from the column names for the test set
# Note: The columns in the test set should exactly match those in the training set
X_test_encoded_df.columns = new_column_names_train

####### 3. Machine Learning Model

#3a. Initiate the classifiers
classifiers = {
    "lr": LogisticRegression(),
    "svc": SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "decisiontree": DecisionTreeClassifier(),
    "bagging": ensemble.BaggingClassifier(),
    "rfc": ensemble.RandomForestClassifier(),
    "adaboost": ensemble.AdaBoostClassifier(),
    "gboost": ensemble.GradientBoostingClassifier(),
    "hgboost": ensemble.HistGradientBoostingClassifier(),
    "gaussnb": GaussianNB(),
    "ann": MLPClassifier(),
}






#3b. Create function to check the model performances
def evaluate_model(y_test, y_pred):
    all_metrics = {
        "acc": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, pos_label='Yes'), 
        "recall": metrics.recall_score(y_test, y_pred, pos_label='Yes'),  
        "f1": metrics.f1_score(y_test, y_pred, pos_label='Yes'),  
    }
    return all_metrics






#3c.Check the model performances and plot the ROC curve

performance_data = []
for name, model in classifiers.items():
    # Fit the model on the encoded training data
    model.fit(X_train_encoded_df, y_train)
    # Predict using the encoded test data
    y_pred = model.predict(X_test_encoded_df)
    performance = evaluate_model(y_test, y_pred)
    # Add model name to the performance dictionary
    performance['model'] = name
    performance_data.append(performance)

# Create a DataFrame from the performance data
model_performance_df = pd.DataFrame(performance_data)
# Reorder the DataFrame to have the model name as the first column
model_performance_df = model_performance_df[['model', 'acc', 'precision', 'recall', 'f1']]

# Convert string labels to binary
label_encoder = LabelEncoder()
y_train_binary = label_encoder.fit_transform(y_train)
y_test_binary = label_encoder.transform(y_test)

# Now use y_train_binary and y_test_binary for training and evaluation
plt.figure(figsize=(10, 8))
for name, model in classifiers.items():
    # Fit the model on the encoded training data
    model.fit(X_train_encoded_df, y_train_binary)
    # Predict probabilities for the positive class using the encoded test data
    y_proba = model.predict_proba(X_test_encoded_df)[:, 1]
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
    roc_auc = auc(fpr, tpr)
    # Plot
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()






#3d. Feature importance analysis
for name, model in classifiers.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"Feature Importances in {name}")
        # Adjust to use the shape and column names from the encoded DataFrame
        plt.bar(range(X_train_encoded_df.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train_encoded_df.shape[1]), X_train_encoded_df.columns[indices], rotation=90)
        plt.tight_layout()
        plt.show()


####### 4. Addressing the class imbalances and check performances again

resampling_strategies = {
    "ADASYN": ADASYN(),
    "KMeansSMOTE": KMeansSMOTE(),
    "SMOTE": SMOTE(),
    "SMOTE-ENN": SMOTEENN(),
    "SMOTE-Tomek": SMOTETomek(),
    "RandomOverSampler": RandomOverSampler(),
    "RandomUnderSampler": RandomUnderSampler()
}
resampling_performance_data = []
for strategy_name, strategy in resampling_strategies.items():
    print(f"Applying {strategy_name}...")
    # Resample the encoded training data
    X_resampled, y_resampled = strategy.fit_resample(X_train_encoded_df, y_train)
    # Evaluate each classifier
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_resampled, y_resampled)
        y_pred = classifier.predict(X_test_encoded_df)
        performance = evaluate_model(y_test, y_pred)
        # Add strategy and classifier names to the performance dictionary
        performance['strategy'] = strategy_name
        performance['classifier'] = classifier_name
        resampling_performance_data.append(performance)

# Create a DataFrame from the resampling performance data
resampling_model_performance_df = pd.DataFrame(resampling_performance_data)
# Reorder the DataFrame
resampling_model_performance_df = resampling_model_performance_df[['strategy', 'classifier', 'acc', 'precision', 'recall', 'f1']]

plt.figure(figsize=(15, 10))
for strategy_name, strategy in resampling_strategies.items():
    print(f"Applying {strategy_name}...")
    # Resample the encoded training data
    X_resampled, y_resampled = strategy.fit_resample(X_train_encoded_df, y_train)
    for classifier_name, classifier in classifiers.items():
        # Fit the classifier
        classifier.fit(X_resampled, y_resampled)
        # Predict probabilities for the positive class using the encoded test data
        y_proba = classifier.predict_proba(X_test_encoded_df)[:, 1]
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        # Plot
        plt.plot(fpr, tpr, label=f'{classifier_name} with {strategy_name} (area = {roc_auc:.2f})')
# Add a 'No Skill' line and labels
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves with Different Resampling Strategies')
plt.legend(loc='lower right')
plt.show()





