#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

# scatter plot matrix

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
        
        plots_amount  = len(cols_list)-1
        fig, axs = plt.subplots(plots_amount, plots_amount, figsize=(25, 15))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.suptitle('Scatter plots ')
        
        for i in range(1, len(cols_list)):
            for j in range(1, len(cols_list)):
                for house in houses_names:
                    house_df = num_df.loc[num_df['Hogwarts House'] == house]
                    if i == j:
                        axs[i-1][j-1].text(0.1, 0.5, cols_list[i], fontsize=11)
                    else:
                        col1 = house_df.columns[i]
                        col2 = house_df.columns[j]
                        axs[i-1][j-1].scatter(house_df.loc[:,col1], house_df.loc[:,col2], \
                            marker='.', color=houses_colors[house], alpha=0.5)


        for ax in axs.flat:
            ax.label_outer()
        plt.show()

        print('Answer: \nAstronomy and Defence Against Dark Arts. \
            \nHistory of Magic and Flying.')