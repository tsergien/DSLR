#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        cols_list = ['Hogwarts House', 'Arithmancy', 'Astronomy', 'Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Care of Magical Creatures', 'Charms', 'Flying']

        num_df = df[cols_list]
        print(num_df.describe())
        for col in num_df.columns:
            plt.hist(num_df.loc[:,col], bins=30, alpha=0.5)
            plt.title(col)
            plt.show()
        
        # ax = num_df.plot.hist(by='Hogwarts House', bins=30, alpha=0.5)
        # plt.show()

