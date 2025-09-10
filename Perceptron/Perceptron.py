import numpy as np


class Perceptron:



def unitStepFunction(self,x):
    return np.where(x>=0 , 1 ,0 )