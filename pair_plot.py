#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys
from scatter_plot import my_scatter

# pair plot matrix

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
            my_scatter(num_df, houses_colors)
            print('Answer: \nHerbology and Defence Against Dark Arts.')
        except Exception as e:
            print(f'Exception: {e}')
