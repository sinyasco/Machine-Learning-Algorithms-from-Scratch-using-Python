import numpy as np


class Perceptron:
  def __init__(self,lr=0.01, n_iters=1000):
    self.lr = 0.01
    self.n_iters=1000
    self.activationFunction = self.unitStepFunction
    self.weights = None 
    self.bias = None 

  def fit(self,X,y):
    n_samples, n_features = X.shape
    #init weights
    self.weights = np.zeros(n_features)
    self.bias = 0
    y_ = np.array([1 if i >0 else 0 for i in y])
    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        linear = np.dot(x_i , self.weights)+self.bias
        y_predicted = self.activationFunction(linear)
        update = self.lr*(y_[idx] - y_predicted)
        self.weights = self.weights + update*x_i
        self.bias = self.bias + update
        
    

  def predict(self,X):
    linearFunction = np.dot(X,self.weights)+self.bias
    y_predicted = self.activationFunction(linearFunction)
    return y_predicted


  def unitStepFunction(self,x):
    return np.where(x>=0 , 1 ,0 )
  

#small data set - OR

X = np.array([
    [0, 0],
    [0, 23],
    [16, 0]
])
y = np.array([0, 1, 1])  

model = Perceptron(lr=0.01, n_iters=10)
model.fit(X, y)
X2 = np.array([[100,1]])
preds = model.predict(X2)
print("Predictions:", preds)