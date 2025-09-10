import numpy as np

class SVM:
    def __init__(self,lr=0.001,lambdaa=0.01,n_iters=1000):
        self.lr = lr
        self.lambdaa = lambdaa
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self,X,y):
        y_ = np.where(y<=0, -1 , 1)
        n_samples , n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                cndt = y_[idx] * np.dot(x_i,self.w)+self.b >=1
                if cndt:
                    self.w -= self.lr * 2*self.lambdaa*self.w
                else:
                    self.w -= self.lr * 2*self.lambdaa*self.w - np.dot(x_i,y_[idx])
                    self.b -= self.lr * y_[idx]

    def predict(self,X):
        linear = np.dot(X, self.w) - self.b
        return np.sign(linear)