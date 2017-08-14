# -*- coding: utf-8 -*-
"""
@author: C
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class model():
    def __init__(self):
        self.data_set = load_iris()
        
    def prepare_data(self):
        """ prepare train and test set """
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
                                            train_test_split(
                                                self.data_set.data, 
                                                self.data_set.target,
                                                test_size=0.3,
                                                random_state=43)
                                                
    def learn_model(self):
        """ create and learn model """
        
        self.k_nei_clf = KNeighborsClassifier()
        self.k_nei_clf.fit(self.X_train, self.y_train)
        print(self.k_nei_clf.score(self.X_test, self.y_test))
    
if __name__ == '__main__':
    ai = model()
    ai.prepare_data()
    ai.learn_model()