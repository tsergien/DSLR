#!/usr/bin/env python3

import pandas as pd
import sys
from Predictor import Predictor
import pickle
import numpy as np

# this script have to generate hpuses.csv (two columns: Index Hogwart House)

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Program need file with test data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")
        num_df = df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
        predictor = pickle.load(open('weights.sav', 'rb'))
        x = num_df.loc[:,['Herbology', 'Defense Against the Dark Arts']]
        y = num_df.loc[:,['Hogwarts House']]
        for i in range(20):
            x0 = np.insert(x.iloc[i,:].values, 0, 1, axis=0)
            print(f'Actual: {y.iloc[i,:].values}, Predicted: {predictor.predict(x0)}')

        

