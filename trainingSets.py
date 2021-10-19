
import numpy as np
from keras.datasets import mnist

class mnistData:
    
    def __init__(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        mnistData.train = clean(train_X, train_y)
        mnistData.test = clean(test_X, test_y)
    
def clean(data_X, data_y):
    res = []
    for X, y in zip(data_X, data_y):
        
        X_clean = np.zeros((1, 28*28))
        for i in range(28):
            X_clean[0, 28*i: 28*i+28] = X[i]/255
        
        y_clean = np.zeros((1, 10))
        y_clean[0][y] = 1 
        
        res.append((X_clean, y_clean))
    return res
    
def accuracy(network, data):
    mistakes = [0]*10
    train_size = len(data.train)
    test_size = len(data.test)
    train_correct = 0
    test_correct = 0
    for X, y in data.train:
        y_pred = network.evaluate_input(X)
        num_pred = np.argmax(y_pred)
        num_true = np.argmax(y)
        if num_true == num_pred:
            train_correct += 1
        else:
            mistakes[num_true] += 1
        
    for X,y in data.test:
        y_pred = network.evaluate_input(X)
        num_pred = np.argmax(y_pred)
        num_true = np.argmax(y)
        if num_true == num_pred:
            test_correct += 1
        else:
            mistakes[num_true] += 1
    
    print(f'Train Accuracy: {train_correct*100/train_size}% \nTest Accuracy: {test_correct*100/test_size}%\n')
    return mistakes

def predictionVote(*votes):
    count = [0]*10
    for vote in votes:
        count[vote] += 1
    return np.argmax(count)
    

def ensembleCheck(network, x, y):
    
    # Creates the small translations
    x_left, x_right = translate(x, (-1,0)), translate(x, (+1,0))
    x_down, x_up = translate(x, (0,-1)), translate(x, (0,+1))
    
    # Evaluates the predictions with the translated inputs
    y_left, y_right = network.evaluate_input(x_left), network.evaluate_input(x_right)
    y_down, y_up = network.evaluate_input(x_down), network.evaluate_input(x_up)
    
    # Finds the prediction with the actual input
    y_norm = network.evaluate_input(x)
    
    # Translates the predictions into numbers
    num_left, num_right = np.argmax(y_left), np.argmax(y_right)
    num_down, num_up = np.argmax(y_down), np.argmax(y_up)
    num_norm = np.argmax(y_norm)
    
    # Finds the final vote
    num_pred = predictionVote(num_left, num_right, num_down, num_up, num_norm)
    
    # Determines the expected output
    num_true = np.argmax(y)
    
    return (num_true == num_pred)

def ensembleAccuracy(network, data):
    mistakes = [0]*10
    train_size = len(data.train)
    test_size = len(data.test)
    train_correct = 0
    test_correct = 0
    for X, y in data.train:
        num_true = np.argmax(y)
        if ensembleCheck(network, X, y):
            train_correct += 1
        else:
            mistakes[num_true] += 1
        
    for X,y in data.test:
        num_true = np.argmax(y)
        if ensembleCheck(network, X, y):
            test_correct += 1
        else:
            mistakes[num_true] += 1
            
    print(f'Train Accuracy: {train_correct*100/train_size}% \n Test Accuracy: {test_correct*100/test_size}%\n')
    return mistakes
            
def translate(x, direction):
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
    
def mainData():
    data = mnistData()
    return data
    
            