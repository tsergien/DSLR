#!usr/bin/env python3

import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from Predictor import Predictor

# omg dont want to scale that things



class LogisticRegression:
    '''Class for training data and graphing results'''
    def __init__(self, epochs=100, l_rate=0.01) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.estimator = Predictor([0, 0, 0])


    def train(self, df: pd.DataFrame, plot=False):
        np.random.seed(7171)
        num_df = df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
        houses_dict = {'Gryffindor': 1, 'Ravenclaw': 2, 'Slytherin': 3, 'Hufflepuff': 4}
        features_df = num_df.loc[:,['Herbology', 'Defense Against the Dark Arts']]
        features_df.insert(loc=0, column='Bias', value=np.zeros(features_df.shape[0]))
        houses_df = num_df.loc[:,'Hogwarts House'].values

        x = features_df.values
        y = [houses_dict[houses_df[i]] for i in range(len(houses_df))]

        # if not plot:
        for epoch in range(self.epochs):
            self.run_epoch(x, y)
        # else: 
        #     self.animated_training(data, df, mus, sigmas)
        
        pickle.dump(self.estimator, open('weights.sav', 'wb'))
        print(f'Resulting loss function: {self.loss_function(x, y)}')
        return        


    def loss_function(self, x: np.ndarray, y: np.array):
        m = len(y)
        return -sum([y[i] * np.log(self.estimator.predict(x[i])) + (1 - y[i]) * np.log(self.estimator.predict(x[i])) for i in range(m)]) / m


    def derivative(self, x: np.ndarray, y: np.array):
        '''returns gradient (vector)'''
        m = len(y)
        updates = np.array([0, 0, 0])
        for j in range(len(updates)):
            updates[j] = sum([(self.estimator.predict(x[i]) - y[i])*x[i][j] for i in range(m)]) / m
        return updates


    def run_epoch(self, x: np.ndarray, y: np.array):
        self.estimator.weights_update(self.derivative(x, y))
        return


    # def graph(self, xg, yg, dots: np.ndarray):
    #     '''Visualizing scattered dots'''
    #     plt.plot(xg, yg)
    #     plt.xlabel('mileage')
    #     plt.ylabel('estimated price')
    #     plt.xlim(xg.min()-1, xg.max()+1)
    #     plt.scatter(dots[:,0], dots[:, 1], marker='x')
    #     plt.title(title)
    #     plt.show()
    #     return


    # def animated_training(self, data: np.ndarray, df: pd.DataFrame, mus, sigmas):
    #     fig, ax = plt.subplots(figsize=(16, 9), dpi=70)
    #     def animate(epoch: int):
    #         self.run_epoch(data)
    #         ax.clear()
    #         plt.title(f'epoch = {epoch}')
    #         ax.set_xlabel('km')
    #         ax.set_ylabel('price')
    #         ax.set_xlim(data.min(axis=0)[0]-1, data.max(axis=0)[0]+1)
    #         ax.set_ylim(-4, 4)
    #         x = np.linspace(start=data.min(axis=0)[0]-1, stop=data.max(axis=0)[0]+1, num=100)
    #         y = self.estimator.predict(x)
    #         line = plt.plot(x, y, label='prediction')
    #         plt.scatter(data[:,0], data[:, 1], label='raw data', marker='x')
    #         plt.legend()
    #         return line,
    #     ani = animation.FuncAnimation(fig, animate, frames=self.epochs, interval=10, blit=False)
    #     plt.show()
    #     for epoch in range(self.epochs):
    #         self.run_epoch(data)
    #     scaled_x = np.linspace(start=data.min(axis=0)[0]-1, stop=data.max(axis=0)[0]+1, num=100)
    #     self.graph(scaled_x, self.estimator.predict(scaled_x), data, 'k', f'Scaled data ({self.epochs})')
    #     x_lin = np.linspace(start=df.min(axis=0)[0]-1, stop=df.max(axis=0)[0]+1, num=100)
    #     y_lin = self.estimator.predict(scaled_x) * sigmas[1] + mus[1]
    #     self.graph(x_lin, y_lin, (np.matrix([df.km, df.price]).T).A, 'b', 'Resulting unscaled prediction')
    #     return


