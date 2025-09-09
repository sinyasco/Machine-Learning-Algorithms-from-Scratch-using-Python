import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        #init mean var priors
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes,n_features))
        self.priors = np.zeros((n_classes))

        for idx, c in enumerate(self.classes):
         X_c = X[y == c]
         self.mean[idx, :] = X_c.mean(axis=0)
         self.var[idx, :] = X_c.var(axis=0)
         self.priors[idx] = X_c.shape[0] / n_samples


    def predict(self,X):
       y_pred = [self._predict(x) for x in X]
       return np.array(y_pred, dtype=int) 
    
    def _predict(self,x):
       posteriors = []
       for idx in range(len(self.classes)):
          prior = np.log(self.priors[idx])
          likelihood = np.sum(np.log(self.pdf(idx,x)))
          posterior = prior + likelihood
          posteriors.append(posterior)
       return self.classes[np.argmax(posteriors)]



    def pdf(self, class_idx, x):
       mean = self.mean[class_idx,:]
       var = self.var[class_idx,:]+0.0001
       numerator = np.exp(- (x - mean) ** 2 / (2 * var))
       denominator = np.sqrt(2 * np.pi * var)
       return numerator / denominator
    
# TEST

# 1. Input X
X = np.array([
    [1, 2],  # class 0
    [2, 3],  # class 0
    [2, 1],  # class 1
    [3, 2]   # class 1
])
y = np.array([0, 0, 1, 1])

# 2. Fit model

model = NaiveBayes()
model.fit(X, y)

# 3. Predict on new points

X_test = np.array([
    [1.5, 2.0],  # closer to class 0
    [3.0, 1.0],  # closer to class 1
])

preds = model.predict(X_test)
print("\nTest samples:\n", X_test)
print("Predicted labels:", preds)
