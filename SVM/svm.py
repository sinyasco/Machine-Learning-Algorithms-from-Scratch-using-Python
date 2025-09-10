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
                    self.w -= self.lr * 2*self.lambdaa*self.w - y_[idx] * x_i
                    self.b -= self.lr * y_[idx]

    def predict(self,X):
        linear = np.dot(X, self.w) + self.b
        return np.sign(linear)
    
# Training data
X_train = np.array([
    [2, 3],
    [4, 5],
    [6, 2],
    [7, 8],
    [8, 5],
    [9, 7],
    [10, 4]
])

# Labels: 1 if sum > 10, else -1
y = np.array([-1, -1, -1, 1, 1, 1, 1])

y_train = np.array([-1, -1, -1, 1, 1, 1, 1])  # -1 if sum<=10, else +1

# Train model
model = SVM(lr=0.001, lambdaa=0.01, n_iters=1000)
model.fit(X_train, y_train)


# TEST DATA (new, unseen points)
X_test = np.array([
    [1, 8],   # sum=9 → should be -1
    [12, 2],  # sum=14 → should be +1
    [3, 4],   # sum=7 → should be -1
    [9, 9],   # sum=18 → should be +1
    [5, 6]    # sum=11 → should be +1
])

y_test = np.array([-1, 1, -1, 1, 1])  # true labels

# Predictions
preds = model.predict(X_test)

print("Test predictions:", preds)
print("True labels:     ", y_test)

# Compute accuracy
accuracy = np.mean(preds == y_test)
print("Accuracy on test data:", accuracy)