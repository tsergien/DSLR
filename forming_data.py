
import numpy as np
import pandas as pd

def form_data(df, subjects_names):
    all_columns = subjects_names.copy()
    all_columns.append('Hogwarts House')
    num_df = df[all_columns].dropna()
    features_df = num_df.loc[:,subjects_names]
    features_df.insert( loc=0, column='Bias', value=(np.zeros(features_df.shape[0])+1) )
    houses_df = num_df.loc[:,'Hogwarts House'].values

    x = features_df.values
    yG = [float(houses_df[i] == 'Gryffindor') for i in range(len(houses_df))]
    yR = [float(houses_df[i] == 'Ravenclaw') for i in range(len(houses_df))]
    yS = [float(houses_df[i] == 'Slytherin') for i in range(len(houses_df))]
    yH = [float(houses_df[i] == 'Hufflepuff') for i in range(len(houses_df))]

    return x, yG, yR, yS, yH

