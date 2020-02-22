#!usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Predictor import Predictor, PredictorMultiClass


class LogisticRegression:
    '''Class for training data and graphing results'''
    def __init__(self, epochs=100, l_rate=0.1) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.est = [Predictor([np.random.random() for j in range(3)]) for i in range(4)]


    def train(self, df: pd.DataFrame, plot=False):
        num_df = df.loc[:,['Herbology', 'Defense Against the Dark Arts', 'Hogwarts House']].dropna()
        features_df = num_df.loc[:,['Herbology', 'Defense Against the Dark Arts']]
        features_df.insert( loc=0, column='Bias', value=(np.zeros(features_df.shape[0])+1) )
        houses_df = num_df.loc[:,'Hogwarts House'].values

        x = features_df.values
        yG = [float(houses_df[i] == 'Gryffindor') for i in range(len(houses_df))]
        yR = [float(houses_df[i] == 'Ravenclaw') for i in range(len(houses_df))]
        yS = [float(houses_df[i] == 'Slytherin') for i in range(len(houses_df))]
        yH = [float(houses_df[i] == 'Hufflepuff') for i in range(len(houses_df))]
        print(f'G = {sum(yG)}, R = {sum(yR)}, S = {sum(yS)}, H = {sum(yH)}')
        # self.miniBatch(x, [yG, yR, yS, yH], 512)
        self.GD(x, [yG, yR, yS, yH])
        # self.SGD(x, [yG, yR, yS, yH])
        self.save_model()
        return


    def loss_function(self, x: np.ndarray, y: np.array, est: Predictor):
        m = len(y)
        return -sum([y[i] * np.log(est.predict(x[i])) + (1 - y[i]) * np.log(est.predict(x[i])) for i in range(m)]) / m


    def GD(self, x: np.ndarray, ys: np.ndarray):
        for i in range(len(ys)):
            y = ys[i]
            for _ in range(self.epochs): 
                self.est[i].weights_update(self.derivative(x, y, self.est[i]))
            print(f'Resulting loss function: {self.loss_function(x, y, self.est[i])}')        
            
    
    def miniBatch(self, x: np.ndarray, ys: np.ndarray, batch_size: int):
        batch_count = int(len(x) / batch_size)
        for i in range(len(ys)):
            y = ys[i]
            for _ in range(self.epochs):
                for  batch_i in range(batch_count):
                    if batch_i < batch_count-1:
                        batch_x = x[batch_i * batch_size:(batch_i+1)*batch_size]
                        batch_y = y[batch_i * batch_size:(batch_i+1)*batch_size]
                    else:
                        batch_x = x[batch_i * batch_size:]
                        batch_y = y[batch_i * batch_size:]
                    self.est[i].weights_update(self.derivative(batch_x, batch_y, self.est[i]))
            print(f'Resulting loss function: {self.loss_function(x, y, self.est[i])}')


    def SGD(self, x: np.ndarray, ys: np.ndarray):
        self.miniBatch(x, ys, 1)


    def derivative(self, x: np.ndarray, y: np.array, est: Predictor):
        '''returns gradient (vector)'''
        m = len(y)
        updates = np.array([0, 0, 0], dtype=float)
        for j in range(len(updates)):
            updates[j] = self.l_rate * sum([(est.predict(x[i]) - y[i])*x[i][j] for i in range(m)]) / m
        return updates


    def save_model(self, filename='weights.sav'):
        w = np.zeros((4, 3))
        for i in range(len(self.est)):
            w[i] = self.est[i].get_weights()
        mult_class = PredictorMultiClass(w, 4)
        pickle.dump(mult_class, open(filename, 'wb'))


