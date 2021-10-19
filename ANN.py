# -*- coding: utf-8 -*-
"""

Homemade Artificial Neural Network program.

"""

import numpy as np
import random

import Activation as ACT
import CostFunction as CF
from trainingSets import *

cost_functionDefault = CF.CE
activationDefault = ACT.sigmoid

class ANN:
    
    def __init__(self, sizes, activation = activationDefault):
        '''
        Parameters
        ----------
        sizes : List of Integers
            len(sizes) = Number of layers in the ANN
            sizes[0] = Number of neurons in the input layer
            sizes[-1] = Number of neurons in the output layer

        Returns
        -------
        None.
        
        '''
        self.sizes = sizes
        self.layers = len(self.sizes)
        self.weight_init()
        self.activation = activation
    
    def weight_init(self):
        '''
        weightMatrices[i] is a sizes[i+1] x sizes[i] matrix.
        The coordinate (k,l) of this matrix is the weight associated between
        l in the ith layer and k in the i+1th layer.
        
        biasses[i] is a sizes[i+1] array (represented as a 1 x sizes[i+1]
        matrix) representing the biasses.
        '''
        
        sizes, L = self.sizes, self.layers
        
        weightMatrices = []
        biasses = []
        
        for i in range(L - 1):
            
            randomMatrix = np.random.normal(0,
                                 1/(np.sqrt(sizes[i])),
                                 (sizes[i+1], sizes[i]))
            
            randomVector = np.random.normal(0,1,(1,sizes[i+1]))
            
            weightMatrices.append(randomMatrix)
            biasses.append(randomVector)
            
        self.weightMatrices = weightMatrices
        self.biasses = biasses
            
    def evaluate_input(self, X):
        '''
        Evaluates the ANN when the input is X.
        
        Parameters
        ----------
        X : Input represented as a 1 x sizes[0] matrix.

        Returns
        -------
        Y: The output computed by the ANN
        
        '''
        L = self.layers
        weightMatrices, biasses = self.weightMatrices, self.biasses
        activation = self.activation
        a = X
        for i in range(L - 1):
            z = a @ weightMatrices[i].T + biasses[i]
            a = activation(z)
        return a
    
    def layers_outputs(self, X):
        '''
        Evaluates the ANN when the input is X. Additionally, it returns the 
        outputs and inputs that every individual layer receives.

        Parameters
        ----------
        X : Input represented as a 1 x sizes[0] matrix.

        Returns
        -------
        A : Output of Layers: A[i] is the output of layer i
        Z : Input of Layers: Z[i] is the input of layer i

        '''
        L = self.layers
        weightMatrices, biasses = self.weightMatrices, self.biasses
        activation = self.activation
        A, Z = [X], [X]
        for i in range(L - 1):
            Z.append( A[-1] @ weightMatrices[i].T + biasses[i] )
            A.append( activation(Z[-1]) )
        return A, Z
    
    def gradient_descent(self, trainSet, learning_rate,
                         cost_function=cost_functionDefault,
                         weight_decay=0, add_noise=None, augment=False):
        '''
        Perform the gradient_descent algorithm in the training set described by
        trainSet.

        Parameters
        ----------
        trainSet : TYPE List of tuples (X,Y). X is a 1 x sizes[0] matrix and Y is a
        1 x sizes[L-1] matrix. 
            DESCRIPTION. The training set. Ideally, ANN(X) = Y. 
            
        learning_rate : TYPE Positive real number.
            DESCRIPTION. The learning rate.
            
        cost_function : TYPE CostFunction type.
            DESCRIPTION. The cost function used by the GD algorithm.
            
        weight_decay : TYPE non-negative real number, optional 
            DESCRIPTION. The L2 regularization weight-decay. The default is 0.

        Returns
        -------
        None.

        '''
        
        weightMatrices, biasses = self.weightMatrices, self.biasses
        activation = self.activation
        L = self.layers
        m = len(trainSet) 
        
        # if augment:
        #     m*=5 # Account for all translations
            
        alpha = learning_rate/m
        
        weightGradient = [np.zeros(W.shape) for W in weightMatrices]
        biasGradient = [np.zeros(b.shape) for b in biasses]
        
        # if augment:
        #     newtrainSet = []
        #     for X,Y in trainSet:
        #         translationList = [translate(X, direction) for\
        #                            direction in [(0,0), (-1,0), (+1,0), (0,-1), (0,+1)]]
        #         newtrainSet += [(x,Y) for x in translationList]
        # else:
        #     newtrainSet = trainSet
        
        newtrainSet = trainSet
        
        for X, Y in newtrainSet:
            # if add_noise is not None:
            #     X = X + np.random.normal(0.1, add_noise, X.shape)
            
            A, Z = self.layers_outputs(X)
            y_predicted = A[L-1]
            output_gradient = cost_function.gradient(np.clip(y_predicted, 1e-3, 1 - 1e-3),
                                                     Y)
            delta = [output_gradient * activation.derivative(Z[L-1])]
            
            #Backpropagation
            for i in range(L-2):
                #Compute deltas
                
                P = weightMatrices[L-i-2] @ np.diag(activation.derivative(Z[L-i-2])[0])
                delta.append( delta[i] @ P )
                             
            for i in range(L-1):
                #Update the Gradients
                biasGradient[L-i-2] = biasGradient[L-i-2] + delta[i] 
                
                P = delta[i].T @ A[L-i-2]
                weightGradient[L-i-2] = weightGradient[L-i-2] + P
                
                #Weight Decay
                Q = weight_decay*weightMatrices[L-i-2]
                weightGradient[L-i-2] = weightGradient[L-i-2] + Q
        
        for i in range(L-1):
            self.biasses[i] = self.biasses[i] - alpha*biasGradient[i]
            self.weightMatrices[i] = self.weightMatrices[i] - alpha*weightGradient[i]
            
         
    def SGD(self, trainSet, learning_rate, sizeMiniBatch, numMiniBatch,
            epochs, cost_function=cost_functionDefault, weight_decay=0,
            showProgress=True, add_noise=None, augment=False):
        '''
        Performs the SGD algorithm, by changing the weights.

        Parameters
        ----------
        trainSet : TYPE
            DESCRIPTION.
        learning_rate : TYPE
            DESCRIPTION.
        sizeMiniBatch : TYPE
            DESCRIPTION.
        numMiniBatch : TYPE
            DESCRIPTION.
        epochs : TYPE
            DESCRIPTION.
        cost_function : TYPE, optional
            DESCRIPTION. The default is cost_functionDefault.
        weight_decay : TYPE, optional
            DESCRIPTION. The default is 0.
        showProgress : TYPE, optional
            DESCRIPTION. The default is True.
        add_noise : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        for epoch in range(epochs):
            
            miniBatches = [random.sample(trainSet, sizeMiniBatch) for _ in range(numMiniBatch)]
            
            for miniBatch in miniBatches:
                self.gradient_descent(miniBatch, learning_rate, cost_function,
                                      weight_decay, add_noise, augment)
                
            if showProgress:
                cost = self.mean_cost(trainSet, cost_function)
                print(f'\n Epoch {epoch+1} out of {epochs}. Cost: {cost} \n')
        
    def mean_cost(self, evalSet, cost_function=cost_functionDefault):
        m = len(evalSet)
        cost = 0
        for X,Y in evalSet:
            y_predicted = self.evaluate_input(X)
            cost += cost_function(np.clip(y_predicted,1e-3, 1-1e-3), Y)
        return cost/m
    
    ## TODO
    def save_network(self, fileBias, fileWeight):
        np.save(fileBias, self.biasses)
        np.save(fileWeight, self.weightMatrices)
    
    def load_network(self, fileBias, fileWeight):
        self.biasses = np.load(fileBias)
        self.weightMatrices = np.load(fileWeight)
        self.layers = len(self.sizes)
        self.sizes = [self.weightMatrices[0].shape[1]]
        self.sizes = self.sizes + [self.biasses[i,0,:].size for i in range(self.layers-1)]