#!/usr/bin/env python3

import pandas as pd
import sys
from LogisticRegression import LogisticRegression
import numpy as np
import pickle

#train a logistic regressision for each house  to predict probability of aiming to this class

def train_test_split(df: pd.DataFrame, test_frac=0.2):
    rand_ind = np.random.permutation(df.shape[0])
    test_size = int(df.shape[0] * test_frac)
    test_df = df.loc[rand_ind[:test_size]]
    train_df = df.loc[rand_ind[test_size:]]
    return train_df, test_df


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        np.random.seed(777)
        df = pd.read_csv(sys.argv[1], sep=",")
        train_df, test_df = train_test_split(df, 0.1)
        
        regressor = LogisticRegression()
        regressor.train(df)
    

    
