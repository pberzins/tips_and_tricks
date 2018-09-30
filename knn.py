import numpy as np
from sklearn import datasets


class KNearestNeighbors:
    def __init__(self, k, distance):
        """Inputs self, k and type of distance
        """
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """ X is the feature matrix as a 2d numpy Arrays
        y is the labels as a numpy array
        """
        self.data = X
        self.labels = y

    def predict(self, X_new):
        """Takes in a data kind similiar to data set
            will return a numpy array of predicted labels
        """
        zero = np.zeros(len(X_new))
        hello = X_new.reshape(len(X_new), 1, 4)
        for e in range(0, len(X_new)):
            eucl = np.sqrt(np.sum((self.data - X_new[e])**2, axis=1))
            zero[e] = np.mean(self.labels[np.argsort(eucl, axis=0)][:self.k])
        return zero

#eucl = np.sqrt(np.sum((self.data - hello)**2,axis=1))
# return self.labels[np.argsort(eucl,axis = 0)]

    def score(self):
        #all = self.labels[np.argsort(self.labels)]
        #only_0 = all[x==0]
        pass
