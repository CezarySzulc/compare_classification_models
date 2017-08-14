# -*- coding: utf-8 -*-
"""
@author: C
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class model():
    def __init__(self):
        self.data_set = load_iris()
        print(self.data_set.target)
        print(type(self.data_set))
        
    def prepare_data(self):
        """ prepare train and test set """
        
        X_train, X_test, y_train, y_test = train_test_split(
                                                self.data_set.data, 
                                                self.data_set.target,
                                                test_size=0.3,
                                                random_state=43)
                                                
    def learn_model(self):
        """ """
        pass
    
if __name__ == '__main__':
    ai = model()