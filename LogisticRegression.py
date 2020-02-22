#!usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Predictor import Predictor, PredictorMultiClass


np.random.seed(777)

class LogisticRegression:
    '''Class for training data and graphing results'''
    def __init__(self, epochs=5, l_rate=0.1) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.est = [Predictor([np.random.random() for j in range(3)]) for i in range(4)]



    def train(self, df: pd.DataFrame, plot=False):
        np.random.seed(7171)
        num_df = df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
        features_df = num_df.loc[:,['Herbology', 'Defense Against the Dark Arts']]
        features_df.insert( loc=0, column='Bias', value=(np.zeros(features_df.shape[0])+1) )
        houses_df = num_df.loc[:,'Hogwarts House'].values


        x = features_df.values
        yG = [houses_df[i] == 'Gryffindor' for i in range(len(houses_df))]
        yR = [houses_df[i] == 'Ravenclaw' for i in range(len(houses_df))]
        yS = [houses_df[i] == 'Slytherin' for i in range(len(houses_df))]
        yH = [houses_df[i] == 'Hufflepuff' for i in range(len(houses_df))]

        i = 0
        for y in [yG, yR, yS, yH]:
            for _ in range(self.epochs):
                self.est[i].weights_update(self.derivative(x, y, i))
            filename = 'weights' + str(i) + '.sav'
            pickle.dump(self.est[i], open(filename, 'wb'))
            print(f'Resulting loss function: {self.loss_function(x, y, i)}')
            i = i + 1
        self.save_model()
        return


    def loss_function(self, x: np.ndarray, y: np.array, est_i):
        m = len(y)
        return -sum([y[i] * np.log(self.est[est_i].predict(x[i])) + (1 - y[i]) * np.log(self.est[est_i].predict(x[i])) for i in range(m)]) / m


    def derivative(self, x: np.ndarray, y: np.array, est_i):
        '''returns gradient (vector)'''
        m = len(y)
        updates = np.array([0, 0, 0])
        for j in range(len(updates)):
            updates[j] = self.l_rate * sum([(self.est[est_i].predict(x[i]) - y[i])*x[i][j] for i in range(m)]) / m
        return updates


    def save_model(self, filename='weights'):
        w = np.zeros(4, 3)
        for i in range(len(self.est)):
            w[i] = self.est[i].get_weights()
        mult_class = PredictorMultiClass(w)
        pickle.dump(mult_class, open(filename, 'w'))


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
    #         y = self.est.predict(x)
    #         line = plt.plot(x, y, label='prediction')
    #         plt.scatter(data[:,0], data[:, 1], label='raw data', marker='x')
    #         plt.legend()
    #         return line,
    #     ani = animation.FuncAnimation(fig, animate, frames=self.epochs, interval=10, blit=False)
    #     plt.show()
    #     for epoch in range(self.epochs):
    #         self.run_epoch(data)
    #     scaled_x = np.linspace(start=data.min(axis=0)[0]-1, stop=data.max(axis=0)[0]+1, num=100)
    #     self.graph(scaled_x, self.est.predict(scaled_x), data, 'k', f'Scaled data ({self.epochs})')
    #     x_lin = np.linspace(start=df.min(axis=0)[0]-1, stop=df.max(axis=0)[0]+1, num=100)
    #     y_lin = self.est.predict(scaled_x) * sigmas[1] + mus[1]
    #     self.graph(x_lin, y_lin, (np.matrix([df.km, df.price]).T).A, 'b', 'Resulting unscaled prediction')
    #     return


