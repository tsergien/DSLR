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
        # houses = ['Ravenclaw', 'Gryffindor', 'Slytherin', 'Hufflepuff']

        num_df = df[cols_list]
        # print(num_df.describe())
        # for col in num_df.columns:
        #     plt.hist(num_df.loc[:,col], bins=30, alpha=0.5)
        #     plt.title(col)
        #     plt.show()
        
        fig, axs = plt.subplots(len(cols_list), len(cols_list))
        for i in range(len(cols_list)):
            for j in range(len(cols_list)):
                if i != j:
                    col = num_df.columns[i]
                    axs[i][j].hist(num_df.loc[:,col], bins=30, alpha=0.5)
                    axs[i][j].set_title(col)
        
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

        # ax = num_df.plot.hist(by='Hogwarts House', bins=30, alpha=0.5)
        # plt.show()

