#!/usr/bin/env python3

import pandas as pd
import sys
from Predictor import PredictorMultiClass
from LogisticRegression import LogisticRegression
import pickle
import numpy as np


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
        train_df, test_df = train_test_split(df, 0.2)
        
        regressor = LogisticRegression()
        regressor.train(train_df)



        num_df = test_df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
        predictor = pickle.load(open('weights.sav', 'rb'))

        x = num_df.loc[:,['Herbology', 'Defense Against the Dark Arts']]
        x.insert( loc=0, column='Bias', value=(np.zeros(x.shape[0])+1) )
        y = num_df.loc[:,['Hogwarts House']]
        hd = {'Gryffindor':1, 'Ravenclaw':2, 'Slytherin':3, 'Hufflepuff':4}
        y_pred = np.zeros(y.shape[0])
        y_true = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            y_pred[i] = predictor.predict(x.iloc[i,:].values)
            y_true[i] = hd[y.iloc[i,0]]
        true_answers = sum( [y_true[i] == y_pred[i] for i in range(len(y_pred))] )
        print(f'{true_answers} true answers of {y.shape[0]}')
        print(f'My accuracy score: {true_answers / y.shape[0]}')

