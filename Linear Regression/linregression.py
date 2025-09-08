import numpy as np

class LinearRegression:

    def __init__(self,learningRate = 0.001,n_iters=1000):
        self.learningRate = learningRate
        self.n_iters = n_iters
        self.weight = None 
        self.bias = None 

    def fit(self,X,y):
        n_samples , n_features = X.shape # number of samples is number of rows, number of features is number of columns, X.shape returns rowsNb, columnsNb
        self.weight = np.zeros(n_features) # number of features = dimension of weight vector , np.zeros returns a 1d vector
        self.bias = 0

        for _ in range(self.n_iters) :
         yPredicted = np.dot(X, self.weight) + self.bias 
         dw = (1/n_samples) * np.dot(X.T , (yPredicted-y))
         db = (1/n_samples) * np.sum(yPredicted-y)
         self.weight = self.weight - self.learningRate * dw 
         self.bias = self.bias - self.learningRate * db


    
    def predict(self,X):
        yPredicted = np.dot(X , self.weight) + self.bias
        return yPredicted
    
#test
X = np.array([[1], [3], [4], [6], [5]])
X2 = np.array([[3], [4], [0], [20], [3], [8]])

y = np.array([2, 4, 6, 8, 10])

# create & train model
model = LinearRegression(learningRate=0.1, n_iters=1000)
model.fit(X, y)

# Predictions
preds = model.predict(X)
print("Predictions:", preds)
print("Weights:", model.weight)
print("Bias:", model.bias)

