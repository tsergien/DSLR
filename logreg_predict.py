#!/usr/bin/env python3

import pandas as pd
import sys
from Predictor import PredictorMultiClass
import pickle
import numpy as np


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Program need file with test data and a file with weights. Please, pass it as argument.")
    else:
        try:
            df = pd.read_csv(sys.argv[1], sep=",")
            num_df = df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
            predictor = pickle.load(open(sys.argv[2], 'rb'))

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

            rev = {1 : 'Gryffindor', 2 : 'Ravenclaw', 3 : 'Slytherin', 4 : 'Hufflepuff'}
            y_pred_str = [rev[y_pred[i]] for i in range(len(y_pred))]
            prediction = pd.DataFrame(y_pred_str, columns=['Hogwarts House'])
            prediction.to_csv('houses.csv', index_label='Index')

        except Exception as e:
            print(f'oops... something went wrong : {e}')
 

