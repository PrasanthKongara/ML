from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        print("t")
        self.f_train=features
        self.l_train=labels
        return
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        raise NotImplementedError

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        r=len(features)
        pred=np.zeros(r,dtype=int)
        for i in range(r):
            count=np.sum(self.get_k_neighbors(features[i]))
            if count>=self.k/2:
                pred[i]=1
            else:
                pred[i]=0
        return pred
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        raise NotImplementedError

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:    
        le=len(self.f_train)
        dist=np.zeros(le)
        l=np.zeros(self.k,dtype=int)
        for i in range(le):
            dist[i]=self.distance_function(point,self.f_train[i])
        ind=dist.argsort()[:self.k]
        for i in range(self.k):
            l[i]=self.l_train[ind[i]]
        return l


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
