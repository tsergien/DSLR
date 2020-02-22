#!/usr/bin/env python3

import numpy as np


class Predictor:
    def __init__(self, weights) -> None:
        self.weights = np.array(weights)
        if weights == None:
            self.weights = np.array([np.random.random() for j in range(3)])


    @staticmethod
    def sigmoid(z: float):
        return 1 / (1 + np.exp(-z))


    def predict(self, x: np.array):
        return self.sigmoid(np.dot(self.weights, x))
    
    
    def get_weights(self):
        return self.weights


    def weights_update(self, updates: np.array) -> None:
        self.weights = self.weights - updates



class PredictorMultiClass:
    def __init__(self, weights: np.ndarray, k=1) -> None:
        '''
        params: 
            k - number of classes
            weights: matrix, each row is weights for a class
        '''
        self.k = k
        self.weights = weights
        # if all(weights) and len(weights.shape) > 1:
        #     self.k = weights.shape[0]


    @staticmethod
    def sigmoid(z: float):
        return 1 / (1 + np.exp(-z))


    def predict(self, x: np.array):
        '''
        Returns 0/1 if regression is simple.
        Returns 1/2.../k if multi class regression.
        '''
        if self.k == 1:
            return int(self.sigmoid(np.dot(self.weights[0], x)) > 0.5)
        else:
            probabilities = np.zeros(self.k, dtype=float)
            for i in range(len(probabilities)):
                probabilities[i] = self.sigmoid(np.dot(self.weights[i], x))
            return np.argmax(probabilities)+1


    def get_weights(self):
        return self.weights

  
    def set_weights(self, weights):
        if weights.shape[0] == self.k and weights.shape[1] == 3:
            self.weights = weights
        else:
            self.weights = np.zeros((self.k, 3))

    def weights_update(self, updates: np.array) -> None:
        self.weights = self.weights - updates


