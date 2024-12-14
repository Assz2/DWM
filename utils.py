import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function removes the columns from train that are not in test
def intersect_features(train_set, test_set):
    smaller_train = train_set[train_set.columns.intersection(test_set.columns)]
    smaller_train = smaller_train.drop(columns=['id'], axis=1)
    return smaller_train

# This function removes the columns from train that have more than a fixed% of NaN
def drop_columns(df, threshold):
    minimum_non_NaN = len(df) * threshold   
    dropped_columns = df.columns[df.isnull().sum() > (len(df) - minimum_non_NaN)].tolist()
    new_df = df.drop(columns=dropped_columns)
    return new_df, dropped_columns

# Extract numerical and categorical columns
def extract_numerical_cathegorical(df):
    numerical = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    return numerical, categorical

# Fill NaN with mean
def convert_float_fill_mean(df):
    numerical_features, categorical_features = extract_numerical_cathegorical(df)
    df_decoded = pd.get_dummies(df, columns=categorical_features)
    df_decoded *= 1
    df_decoded = df_decoded.astype('float64')
    df_floated_filled = df_decoded.fillna(df_decoded.mean())
    return df_floated_filled

def count_feature_clusters(feature_list, feature_clusters):
    #for cluster in feature_clusters:
        #print(f"Cluster {cluster}: {len([col for col in df.columns if cluster in col])}")
    return {cluster: len([col for col in feature_list if cluster in col]) for cluster in feature_clusters}  

# Drop all the records with a fixed % of NaN
def drop_rows(df, threshold):
    minimum_non_NaN = len(df.columns) * threshold
    new_df = df.dropna(thresh=minimum_non_NaN)
    return new_df

def remove_outliers_iqr(df, droppable_columns):
    features = df.drop(columns=droppable_columns, axis=1) 
    
    threshold = 3.0
    
    for col in features.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        features = features[(features[col] >= lower_bound) & (features[col] <= upper_bound)]
    cleaned_df = df.loc[features.index]
    
    return cleaned_df


#tried to remove outliers with zscore but it is not working
def remove_outliers_zscore(df, droppable_columns): 
    features = df.drop(columns=droppable_columns, axis=1)

    threshold = 3

    z_scores = (features - features.mean()) / features.std()
    filtered_rows = (np.abs(z_scores) < threshold).any(axis=1)
    cleaned_df = df.loc[filtered_rows.index]

    return cleaned_df

    