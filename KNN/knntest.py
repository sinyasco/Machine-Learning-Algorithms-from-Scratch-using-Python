import numpy as np  # type: ignore
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        #calculate distances
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        #get k nearest samples,labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #maojrity_vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
# TEST

# Sample training data
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 8],
    [8, 8]
])

y_train = np.array([0, 0, 0, 1, 1, 1])  

# New data to classify
X_test = np.array([
    [1, 2],
    [8, 7]
])

model = KNN(k=3)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions)

