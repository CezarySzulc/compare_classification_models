# -*- coding: utf-8 -*-
"""
@author: C
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

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
        
        param_grid = {'n_neighbors': np.arange(3 ,5)}        
        self.k_nei_clf = GridSearchCV(KNeighborsClassifier(),
                                      param_grid=param_grid)
        self.k_nei_clf.fit(self.X_train, self.y_train)
        print(self.k_nei_clf.score(self.X_test, self.y_test))
        print(self.k_nei_clf.best_params_)
        
        self.log_reg_clf = OneVsRestClassifier(LogisticRegression())
        self.log_reg_clf.fit(self.X_train, self.y_train)
        print(self.log_reg_clf.score(self.X_test, self.y_test))
    
if __name__ == '__main__':
    ai = model()
    ai.prepare_data()
    ai.learn_model()