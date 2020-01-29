#!/usr/bin/env python3

import sys
import pandas as pd
import math
import numpy as np


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        print(df.describe())

        num_df = df.select_dtypes(include=['int', 'float64'])       
        index_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'unique']
        describe_df = pd.DataFrame(0, index=index_list, columns=list(num_df.columns.values)) 
        
        describe_df.loc['count',:] = num_df.shape[0] * np.ones(len(describe_df.loc['count',:])) - num_df.isna().sum()
        describe_df.loc['mean',:] = np.divide(num_df.sum(axis=0, skipna=True), describe_df.loc['count',:])
        squared_sum = np.nansum(np.power([num_df.loc[:,column]-describe_df.loc['mean',column] for column in num_df.columns], 2), axis=1)
        describe_df.loc['std',:] = np.sqrt(np.divide(squared_sum, describe_df.loc['count',:]) )
        describe_df.loc['min',:] = num_df.min(axis=0, skipna=True)
        describe_df.loc['max',:] = num_df.max(axis=0, skipna=True)
        # describe_df.loc['unique',:] = num_df.value_counts()

        rank25 = (np.round(.25 * (describe_df.loc['count',:]+1 ))).astype(int)
        rank50 = (np.round(.5 * (describe_df.loc['count',:]+1 ))).astype(int)
        rank75 = (np.round(.75 * (describe_df.loc['count',:]+1 ))).astype(int)

        for column in num_df.columns:
            subject_score_order = np.array(num_df.loc[:,column]).argsort()
            ranks = subject_score_order.argsort()+1
            describe_df.loc['25%',column] = num_df.iloc[np.where(ranks == rank25[column])[0][0]][column]
            describe_df.loc['50%',column] = num_df.iloc[np.where(ranks == rank50[column])[0][0]][column]
            describe_df.loc['75%',column] = num_df.iloc[np.where(ranks == rank75[column])[0][0]][column]
           

        print(describe_df)

