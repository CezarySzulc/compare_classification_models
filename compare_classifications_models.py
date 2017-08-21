# -*- coding: utf-8 -*-
"""
@author: C
"""

import numpy as np

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
                        classification_report, 
                        confusion_matrix)


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
                                                
    def scatter_matrix(self):
        """ display scatter matrix """
        
        target = pd.DataFrame(self.data_set.target, columns=['flower'])
        df = pd.DataFrame(self.data_set.data, 
                          columns=self.data_set.feature_names)
        color_map = {
            0: "black", 
            1: "yellow", 
            2: "blue"
        }
        colors = target['flower'].map(lambda x: color_map.get(x))
        pd.tools.plotting.scatter_matrix(df, color=colors)
    
    def learn_model(self):
        """ create and learn model """
        
        param_grid = {'n_neighbors': np.arange(1,10)}        
        self.k_nei_clf = GridSearchCV(KNeighborsClassifier(),
                                      param_grid=param_grid)
        self.k_nei_clf.fit(self.X_train, self.y_train)
        
        
        self.log_reg_clf = OneVsRestClassifier(LogisticRegression())
        self.log_reg_clf.fit(self.X_train, self.y_train)
        
        
        print(self.k_nei_clf.predict_proba(self.X_test))
        predict = self.k_nei_clf.predict(self.X_test)
        print(classification_report(self.y_test, predict))
        print(confusion_matrix(self.y_test, predict))
        print(self.k_nei_clf.best_params_)
        
        
        pr_proba = np.array(self.log_reg_clf.predict_proba(self.X_test))
        print(np.sum(pr_proba, axis=1))
        predict = self.log_reg_clf.predict(self.X_test)
        print(classification_report(self.y_test, predict))
        print(confusion_matrix(self.y_test, predict))
    
if __name__ == '__main__':
    ai = model()
    ai.prepare_data()
    ai.learn_model()
    ai.scatter_matrix()