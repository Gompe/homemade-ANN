# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:07:55 2021

@author: gomes
"""

import numpy as np
import random

import Activation as ACT
import CostFunction as CF
import trainingSets as TS

from ANN import *

def randomDataGen(net, size=60000):
    randomData = []
    for _ in range(size):
        x = np.random.uniform(size=(1,28*28))
        y = net.evaluate_input(x)
        randomData.append((x,y))
    return randomData

def mnistDataGen():
    data = TS.main()
    dataTrain = data.train
    return dataTrain

def testNet(net):
    print("Normal Accuracy: \n")
    TS.accuracy(net)
    print("Ensemble Accuracy: \n")
    TS.ensembleAccuracy(net)

def trainingMessage(numNow, numTotal, state, net=None):
    if state == 0:
        print(f"Starting step {numNow}/{numTotal} -- \n")
    if state == 1:
        print(f"Step {numNow}/{numTotal} completed with success --\n")
        print("Evaluations:\n")
        testNet(net)
        
def doubleTraining(sizes, show=False):
    netTeacher = trainingRoutine(sizes, show)
    
    teacherData = randomDataGen(netTeacher)
    del netTeacher
    
    netStudent = trainingRoutine(sizes, show, teacherData)
    
    return netStudent

def trainingRoutine(sizes, show=False, dataTrain=mnistDataGen()):
    
    print("Data Collected -- Success\n")
    
    net = ANN(sizes)
    print("Neural Net Created -- Success\n")
    
    trainingMessage(1,6,0)
    net.SGD(dataTrain, 2, 30, 30, 100, showProgress=show, add_noise=None)
    trainingMessage(1,6,1, net)
    
    trainingMessage(2,6,0)
    net.SGD(dataTrain, 1, 30, 30, 100, showProgress=show, add_noise=None)
    trainingMessage(2,6,1, net)
        
    trainingMessage(3,6,0)
    net.SGD(dataTrain, 0.5, 30, 30, 30, showProgress=show, add_noise=None, augment=True)
    trainingMessage(3,6,1, net)
    
    trainingMessage(4,6,0)
    net.SGD(dataTrain, 0.25, 30, 30, 30, showProgress=show, add_noise=None, augment=True)
    trainingMessage(4,6,1, net)
    
    trainingMessage(5,6,0)
    net.SGD(dataTrain, 0.125, 30, 30, 30, showProgress=show, add_noise=None, augment=True)
    trainingMessage(5,6,1, net)
    
    trainingMessage(6,6,0)
    net.SGD(dataTrain, 0.05, 30, 30, 30, showProgress=show, add_noise=None, augment=True)
    trainingMessage(6,6,1, net)
    
    return net
    

def translate(x, direction):
    if direction == (0,0):
        return x
    xout = np.zeros((1, 28*28))
    if direction == (0,+1): # Up
        xout[0, :-28] = x[0, 28:]
    if direction == (0,-1): # Down
        xout[0, 28:] = x[0, :-28]
    if direction == (-1, 0): #Left
        xout[0, :-1] = x[0, 1:]
        xout[0, 27::28] = 0
    if direction == (+1,0): #Right
        xout[0, 1:] = x[0, :-1]
        xout[0, ::28] = 0
    return xout
