#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

# Which Hogwarts course has a homogeneous score distribution between all four houses

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
        fig, axs = plt.subplots(rows_amount, columns_amount, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle('Scores distributions between all houses')
        
        for i in range(len(cols_list)):
            col = num_df.columns[i]
            axs[i % rows_amount][i % columns_amount].hist(num_df.loc[:,col], bins=30, alpha=0.5, color='b')
            axs[i % rows_amount][i % columns_amount].set_title(col)
        fig.delaxes(axs[rows_amount-1][columns_amount-1])

        for ax in axs.flat:
            ax.label_outer()
        plt.show()


