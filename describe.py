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
# max

if __name__ == "__main__":
    if (len(sys.argv) < 1):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        print(df.describe())

        num_df = df.select_dtypes(include=['int', 'float64'])       
        index_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        describe_df = pd.DataFrame(0, index=index_list, columns=list(num_df.columns.values)) 
        
        describe_df.loc['count',:] = num_df.shape[0] * np.ones(len(describe_df.loc['count',:])) - num_df.isna().sum()
        describe_df.loc['mean',:] = np.divide(num_df.sum(axis=0, skipna=True), describe_df.loc['count',:])
        squered_sum = np.nansum(np.power([num_df.loc[:,column]-describe_df.loc['mean',column] for column in num_df.columns], 2), axis=1)
        describe_df.loc['std',:] = np.sqrt(np.divide(squered_sum, describe_df.loc['count',:]) ) # i have just for like 0.01 less than actual describe... why ?
        describe_df.loc['min',:] = num_df.min(axis=0, skipna=True)
        describe_df.loc['max',:] = num_df.max(axis=0, skipna=True)

        rank25 = np.round(25 / 100 * (describe_df.loc['count',:] + 1)).astype(int)
        rank50 = np.round(50 / 100 * (describe_df.loc['count',:] + 1)).astype(int)
        rank75 = np.round(75 / 100 * (describe_df.loc['count',:] + 1)).astype(int) # its arrays !!! rank for each feature, stupid
        # print(f'ranks: 25={rank25} type {type(rank25)}')

        for column in num_df.columns:
            subject_score_order = np.array(num_df.loc[:,column]).argsort() # argsort creates an array: index of element in arr if arr was sorted
            ranks = subject_score_order.argsort() # and this is array of 1600 elements

            # print(f'ranks = {ranks[3]}, type {type(ranks)}')
            describe_df.loc['25%',column] = ranks[rank25[column]]
            # describe_df.loc['50%',column] = ranks[rank50]
            # describe_df.loc['75%',column] = ranks[rank75]

        print(describe_df)

