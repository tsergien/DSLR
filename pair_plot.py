#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sys

# pair plot matrix


def my_pair_plot(num_df: pd.DataFrame, houses_colors):
    '''
    Accepts DataFrame with numerical features and draws pair plot
    '''
    plots_amount  = num_df.shape[1] - 1
    fig, axs = plt.subplots(plots_amount, plots_amount, figsize=(30, 30))
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.suptitle('Scatter plots ')

    font = {'family' : 'DejaVu Sans',
          'weight' : 'light',
          'size'   : 6}
    matplotlib.rc('font', **font)
    for i in range(1, len(num_df.columns)):
        for j in range(1, len(num_df.columns)):
            for house in houses_colors.keys():
                house_df = num_df.loc[num_df['Hogwarts House'] == house]    
                col_x = house_df.columns[i]
                col_y = house_df.columns[j]
                if i == j:
                    axs[i-1][j-1].hist(house_df.loc[:,col_x], \
                        bins=30, alpha=0.3, color=houses_colors[house])
                else:
                    axs[i-1][j-1].scatter(house_df.loc[:,col_x], house_df.loc[:,col_y], \
                        marker='.', color=houses_colors[house], alpha=0.5)
                
            if axs[i-1][j-1].is_last_row():
                axs[i-1, j-1].set_xlabel(col_y.replace(' ', '\n'))
                axs[i-1, j-1].tick_params(labelbottom=False)
            if axs[i-1][j-1].is_first_col():
                axs[i-1, j-1].set_ylabel(col_x.replace(' ', '\n'))
                axs[i-1, j-1].tick_params(labelleft=False)
            axs[i-1, j-1].spines['right'].set_visible(False)
            axs[i-1, j-1].spines['top'].set_visible(False)
    for ax in axs.flat:
        ax.label_outer()
    plt.legend(houses_colors.keys(), loc=1, frameon=False, bbox_to_anchor=(1.1, 1.1, 1, 1))
    plt.show()


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        try:
            df = pd.read_csv(sys.argv[1], sep=",")

            cols_list = ['Hogwarts House', 'Arithmancy', 'Astronomy', 'Herbology',
            'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
            'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
            'Care of Magical Creatures', 'Charms', 'Flying']
            houses_colors = {'Gryffindor':'r', 'Ravenclaw':'b', 'Slytherin':'g', 'Hufflepuff':'y'}

            num_df = df[cols_list]
            my_pair_plot(num_df, houses_colors)
            print('Answer: \nHerbology and Defence Against Dark Arts.')
        except Exception as e:
            print(f'Exception: {e}')
