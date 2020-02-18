#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        cols_list = ['Hogwarts House', 'Arithmancy', 'Astronomy', 'Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Care of Magical Creatures', 'Charms', 'Flying']
        houses_names = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
        houses_colors = {'Gryffindor':'r', 'Ravenclaw':'b', 'Slytherin':'g', 'Hufflepuff':'y'}

        num_df = df[cols_list]
        
        rows_amount = 2
        columns_amount = 7
        fig, axs = plt.subplots(rows_amount, columns_amount, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle('Scores distributions between all houses')
        
        for i in range(len(cols_list)):
            for house in houses_names:
                house_df = num_df.loc[num_df['Hogwarts House'] == house]
                col = house_df.columns[i]
                axs[i % rows_amount][i % columns_amount].hist(house_df.loc[:,col], \
                    bins=30, alpha=0.3, color=houses_colors[house])
                axs[i % rows_amount][i % columns_amount].set_title(col)

        for ax in axs.flat:
            ax.label_outer()
        plt.legend(houses_names)
        plt.show()


        answer = ['Care of Magical Creatures']
        for i in range(len(answer)):
            for house in houses_names:
                house_df = num_df.loc[num_df['Hogwarts House'] == house]
                col = answer[i]
                plt.hist(house_df.loc[:,col], \
                    bins=30, alpha=0.4, color=houses_colors[house])
            plt.title(col)
            plt.legend(houses_names)
            plt.show()
