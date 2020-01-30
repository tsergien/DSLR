#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

# What are the two features that are similar ?

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        cols_list = ['Arithmancy', 'Astronomy', 'Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Care of Magical Creatures', 'Charms', 'Flying']

        num_df = df[cols_list]
        
        rows_amount = 2
        columns_amount = 7
        plots_amount  = len(cols_list)
        fig, axs = plt.subplots(plots_amount, plots_amount, figsize=(25, 15))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.suptitle('Scatter plots ')
        
        for i in range(len(cols_list)):
            for j in range(len(cols_list)):
                if i == j:
                    axs[i][j].text(0.1, 0.5, cols_list[i], fontsize=11)
                else:
                    col1 = num_df.columns[i]
                    col2 = num_df.columns[j]
                    axs[i][j].scatter(num_df.loc[:,col1], num_df.loc[:,col2], marker='.')
            

        for ax in axs.flat:
            ax.label_outer()
        plt.show()
