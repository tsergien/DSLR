#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

# scatter plot matrix

def my_scatter(num_df: pd.DataFrame, houses_colors):
    '''
    Accepts DataFrame with numerical features and draws scatter plot
    '''
    plots_amount  = num_df.shape[1]-1
    fig, axs = plt.subplots(plots_amount, plots_amount, figsize=(25, 15))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle('Scatter plots ')
     
    for i in range(1, len(num_df.columns)):
        for j in range(1, len(num_df.columns)):
            for house in houses_colors.keys():
                house_df = num_df.loc[num_df['Hogwarts House'] == house]
                if i == j:
                    axs[i-1][j-1].text(0.1, 0.1, num_df.columns[i].replace(' ', '\n'), fontsize=9)
                else:
                    col1 = house_df.columns[i]
                    col2 = house_df.columns[j]
                    axs[i-1][j-1].scatter(house_df.loc[:,col1], house_df.loc[:,col2], \
                        marker='.', color=houses_colors[house], alpha=0.5)
    for ax in axs.flat:
        ax.label_outer()
    plt.legend(houses_colors.keys(), loc=1, bbox_to_anchor=(1.1, 1.1, 1, 1))
    plt.show()

    


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")
        
        cols_list = ['Hogwarts House', 'Arithmancy', 'Astronomy', 'Herbology',
        'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
        'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
        'Care of Magical Creatures', 'Charms', 'Flying']
        houses_colors = {'Gryffindor':'r', 'Ravenclaw':'b', 'Slytherin':'g', 'Hufflepuff':'y'}
        num_df = df[cols_list]
        my_scatter(num_df, houses_colors)
        
        for house in houses_colors.keys():
            house_df = num_df.loc[num_df['Hogwarts House'] == house]
            plt.scatter(house_df.loc[:,'Astronomy'], house_df.loc[:,'Defense Against the Dark Arts'], \
                marker='.', color=houses_colors[house], alpha=0.5)
        plt.legend(houses_colors.keys())
        plt.xlabel('Astronomy')
        plt.ylabel('Defense Against the Dark Arts')
        plt.show()
        print('Answer: \nAstronomy and Defence Against Dark Arts.')
