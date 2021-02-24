import acquire
import pandas as pd
import numpy as np
from env import host, user, password
from pydataset import data

iris_prac = acquire.get_iris_data()

def clean_iris():
    '''
    prep_iris will take a dataframe acquired as df and remove species_id and 
    measurement_id. The function will then rename the species_name col to 'species'
    Finally, the categorical species name will have dummy values created for them and the
    table will be concatanted to bring it all together as on dataframe
    
    return: single cleaned dataframe
    '''
    
    iris_prac.drop(['species_id', 'measurement_id'], inplace=True)
    iris_prac.rename(columns={"species_name": "species"}, inplace=True)
    dummies = pd.get_dummies(iris_prac[['species']], drop_first=False)
    pd.get_dummies(iris_prac[['species']], drop_first=False)
    return pd.concat([iris_prac, dummies], axis=1)



