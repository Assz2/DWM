import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# This function remuve the columns from train that are not in test
def intersect_features(train_set, test_set):
    smaller_train = train_set[train_set.columns.intersection(test_set.columns)]
    smaller_train = smaller_train.drop(columns=['id'], axis=1)
    return smaller_train

# This function remuve the columns from train that have more than 40% of NaN
def drop_columns(df, threshold = 0.4):
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