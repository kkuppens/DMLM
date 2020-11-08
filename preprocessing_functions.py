import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)

def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def extract_cabin_letter(df, var):
    # captures the first letter
    df[var]= df[var].str.replace("[^ABCDEFGHIJKLMNOPQRSTUVWXYZ'-. ]", "")

def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_na'] = np.where(df[var].isnull(), 1, 0)

def impute_na(df, var, replacement='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)

def remove_rare_labels(df, var, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')

def encode_categorical(df):
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    vars_cat = [var for var in df.columns if df[var].dtypes == 'O']
    return pd.get_dummies(df, columns=vars_cat)


def check_dummy_variables(df, dummy_list):

    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    i= 0
    col_present = np.isin(dummy_list,df.columns)
    for col in col_present:
        if col == False:
            df[dummy_list[i]] = 0
        i = i +1
    return df[dummy_list]

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path) # with joblib probably
    return scaler.transform(df)



def train_model(df, target, output_path):
    # initialise the model
    log_model = LogisticRegression(C=0.0005, random_state=0)

    # train the model
    log_model.fit(df, target)

    # save the model
    joblib.dump(log_model, output_path)

    return None



def predict(df, model):
    # load model and get predictions
    log_model = joblib.load(model)
    return log_model.predict(df)
