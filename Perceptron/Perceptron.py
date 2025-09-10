import numpy as np


class Perceptron:
  def init(self,lr=0.01, n_iters=1000):
    self.lr = 0.01
    self.n_iters=1000
    self.activationFunction = self.unitStepFunction
    self.weights = None 
    self.bias = None 

  def fit(self,X,y):
    

  def predict(self,X):
    linearFunction = np.dot(X,self.weights)+self.bias
    y_predicted = self.activationFunction(linearFunction)
    return y_predicted


  def unitStepFunction(self,x):
    return np.where(x>=0 , 1 ,0 )