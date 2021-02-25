import pandas as pd
import numpy as np
import os

### from acquire.py

from env import host, user, password
from pydataset import data
from acquire import get_connection, new_iris_data, get_iris_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


### Clean the Data ###

def clean_iris():
    '''
    prep_iris will take a dataframe acquired as df and remove species_id and 
    measurement_id. The function will then rename the species_name col to 'species'
    Finally, the categorical species name will have dummy values created for them and the
    table will be concatanted to bring it all together as on dataframe
    
    return: single cleaned dataframe

    '''
    
    df = get_iris_data()
    df['species'] = df.species_name
    dropcols = ['species_id', 'measurement_id', 'species_name']
    df.drop(columns=dropcols, inplace=True)
    dummies = pd.get_dummies(df[['species']], drop_first=False)
    pd.get_dummies(df[['species']], drop_first=False)
    return pd.concat([df, dummies], axis=1)


### Train, Test, Validate


df = clean_iris()
train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)

def impute_mode():
    '''
    impute mode for species
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    train[['species']] = imputer.fit_transform(train[['species']])
    validate[['species']] = imputer.transform(validate[['species']])
    test[['species']] = imputer.transform(test[['species']])
    return train, validate, test

def prep_iris_data():
    df = clean_iris()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    train, validate, test = impute_mode()
    return train, validate, test

  


def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.3, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(df, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test
