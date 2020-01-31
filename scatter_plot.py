#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Program need file with data. Please, pass it as argument.")
    else:
        df = pd.read_csv(sys.argv[1], sep=",")

        houses_names = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
        houses_colors = {'Gryffindor':'r', 'Ravenclaw':'b', 'Slytherin':'g', 'Hufflepuff':'y'}
        
        
        for house in houses_names:
            house_df = df.loc[df['Hogwarts House'] == house]
            col1 = 'Astronomy'
            col2 = 'Defense Against the Dark Arts'
            plt.scatter(house_df.loc[:,col1], house_df.loc[:,col2], \
                marker='.', color=houses_colors[house], alpha=0.5)

        plt.title('Scatter plot')
        plt.xlabel('Astronomy')
        plt.ylabel('Defense Against the Dark Arts')
        plt.show()
