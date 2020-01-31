#!/usr/bin/env python3

import pandas as pd
import sys
from Predictor import Predictor

# this script have to generate hpuses.csv (two columns: Index Hogwart House)

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Program need file with test data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")
        weights = pickle.load(open('weights.sav', 'rb'))
        

