#!/usr/bin/env python

import sys
import pandas as pd
import math
import numpy as np

from tabulate import tabulate

# for each feature:
# count
# mean
# std
# min
# 25 %
# 50 %
# 75 %

if __name__ == "__main__":
    if (len(sys.argv) < 1):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        print(df.describe())

        numeric_df = df.select_dtypes(include=['int', 'float64'])       
        index_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%']
        describe_df = pd.DataFrame(0, index=index_list, columns=list(numeric_df.columns.values)) 
        
        for column in numeric_df.columns:
            describe_df[column]['count'] = numeric_df.shape[0]- numeric_df[column].isna().sum()
            describe_df[column]['mean'] = sum(numeric_df[column]) / len(numeric_df[column]) # without NA ?
            


        print(describe_df)
