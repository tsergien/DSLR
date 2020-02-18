#!/usr/bin/env python3

import pandas as pd
import sys
from LogisticRegression import LogisticRegression

#train a logistic regressision for each house  to predict probability of aiming to this class

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        regressor = LogisticRegression()
        regressor.train(df)
        
        

