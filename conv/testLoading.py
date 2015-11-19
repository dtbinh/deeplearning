import numpy as np
import os


def load_data(path):
    images = np.load(path + 'objectImages.npy')
    labelNames = np.load(path + 'objectLabels.npy')
    
    labels = []
    
    for x in labelNames:
        if x == "aaDrink":
            l = 0
        if x == "blackcup":
            l = 1
        if x == "cherryCoke":
            l = 2
        if x == "chickenSoup":
            l = 3
        if x == "cocaColaLight":
            l = 4
        if x == "fanta":
            l = 5
        if x == "hertogJan":
            l = 6
        if x == "pringles":
            l = 7
        if x == "redBull":
            l = 8
        if x == "shampoo":
            l = 9
        if x == "yellowContainer":
            l = 10
        
        labels.append(l)
        
    return images, labels
    

if __name__ == "__main__":
    
    path = '/home/rik/deeplearning/conv/'
    load_data(path);
    