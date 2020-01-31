#!/usr/bin/env python3

import numpy as np


class Predictor:
    def __init__(self, weights=np.array([0, 0, 0])) -> None:
        self.weights = weights


    @staticmethod
    def sigmoid(z: float):
        return 1 / (1 + np.exp(-z))


    def predict(self, x: np.array):
        return self.sigmoid(np.dot(self.weights, x))


    def get_weights(self):
        return self.weights


    def weights_update(self, updates: np.array) -> None:
        self.weights = self.weights - updates


    def set_scaling_parameters(self, mus, sigmas):
        return

    def get_scaling_parameters(self):
        return
